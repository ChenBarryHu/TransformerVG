import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from _3detr.optimizer import *

# load the json files for train and val set
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))


# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split], 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=args.use_height,
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        use_bert=(args.lang_type=="bert")
    )
    # FIXME: change the num_worker based on the machine type
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataset_num_workers)
    return dataset, dataloader


def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = 0
    best_val_metrics = {
        "epoch": 0,
        "loss": float("inf"),
        "ref_loss": float("inf"),
        "lang_loss": float("inf"),
        "objectness_loss": float("inf"),
        "vote_loss": float("inf"),
        "box_loss": float("inf"),
        "lang_acc": -float("inf"),
        "ref_acc": -float("inf"),
        "obj_acc": -float("inf"),
        "pos_ratio": -float("inf"),
        "neg_ratio": -float("inf"),
        "iou_rate_0.25": -float("inf"),
        "iou_rate_0.5": -float("inf")
    }

    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    last_checkpoint = os.path.join(checkpoint_dir, "checkpoint.tar")
    if not os.path.isfile(last_checkpoint):
        os.makedirs(checkpoint_dir, exist_ok=True)
        return epoch, best_val_metrics

    sd = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    epoch = sd["epoch"]
    args = sd["args"]
    best_val_metrics = sd["best_val_metrics"]
    print(f"Found checkpoint at {epoch}. Resuming.")

    model_no_ddp.load_state_dict(sd["model_state_dict"])
    optimizer.load_state_dict(sd["optimizer_state_dict"])
    print(
        f"Loaded model and optimizer state at epoch {epoch}. Loaded best val metrics so far."
    )
    return epoch, best_val_metrics, args

def get_model(args, dataset_config):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    model = RefNet(
        args=args,
        dataset_config=dataset_config,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained pipeline...")

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model.pth") # used model.pth as it stores the best model
        model.load_state_dict(torch.load(pretrained_path), strict=False)

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False
            
            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False
    
    if args.unfreeze_decoder_last_layeres != 0:
        dec_nlayers = args.dec_nlayers
        unfrozen_dec_nlayers = args.unfreeze_decoder_last_layeres
        for idx in range(dec_nlayers-unfrozen_dec_nlayers, dec_nlayers):
            for param in model.detr.decoder.layers[idx].parameters():
                param.requires_grad = True
    if args.unfreeze_mlp_heads:
        for param in model.detr.mlp_heads.parameters():
            param.requires_grad = True

    # to CUDA
    model = model.cuda()

    return model

def get_num_params(model):
    # return num of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params_total = int(sum([np.prod(p.size()) for p in model_parameters]))

    model_parameters_detr = filter(lambda p: p.requires_grad, model.detr.parameters())
    num_params_detr = int(sum([np.prod(p.size()) for p in model_parameters_detr]))

    model_parameters_lang = filter(lambda p: p.requires_grad, model.sequential[0].parameters())
    num_params_lang = int(sum([np.prod(p.size()) for p in model_parameters_lang]))

    model_parameters_matching = filter(lambda p: p.requires_grad, model.sequential[1].parameters())
    num_params_matching = int(sum([np.prod(p.size()) for p in model_parameters_matching]))

    print(f"num_params_trainable_total is {num_params_total}")
    print(f"num_params_trainable_detr is {num_params_detr}")
    print(f"num_params_trainable_lang is {num_params_lang}")
    print(f"num_params_trainable_matching is {num_params_matching}")

    num_params = {
        "total":num_params_total,
        "detr":num_params_detr,
        "lang":num_params_lang,
        "matching":num_params_matching
    }
    return num_params

def get_num_params_total(model):
    # return num of total parameters
    model_parameters = model.parameters()
    model_total_params = sum(p.numel() for p in model_parameters)
    return model_total_params

def get_solver(args, dataloader):
    model = get_model(args, DC)
    # Use 2 optimizers: one for 3detr and one for the reference part
    print(model)
    optimizers = []
    #Use two separate optimizers for 3detr and reference part.
    if args.use_two_optim:
        optimizer_3detr = build_optimizer(args, model.detr)
        optimizers.append(optimizer_3detr)
        optimizer_reference = optim.Adam(
            [
                {"params": model.sequential[0].parameters()},
                {"params": model.sequential[1].parameters(), "lr": 5e-4},
                {"params": model.sequential[2].parameters(), "lr": 5e-4}
            ],
            lr=args.lr, weight_decay=args.wd)
        optimizers.append(optimizer_reference)
    else:
        params_with_decay = []
        params_without_decay = []
        for name, param in model.detr.named_parameters():
            if param.requires_grad is False:
                continue
            if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)

        if args.filter_biases_wd:
            param_groups = [
                {"params": params_without_decay, "lr": 1e-6, "weight_decay": 0.0},
                {"params": params_with_decay, "lr": 1e-6, "weight_decay": args.weight_decay},
            ]
        else:
            param_groups = [
                {"params": params_with_decay, "lr": 1e-6, "weight_decay": args.weight_decay},
            ]
        # LR and WD hyperparameters taken over from the 3DVG paper.
        # Parameters Detection Head
        # param_groups.append(
        #     {"params": model.sequential[0].parameters(),
        #      "lr": 1e-3, "weight_decay": 1e-5}
        # )
        # Parameters Lang Module
        param_groups.append(
            {"params": model.sequential[0].parameters(),
             "lr": 5e-4, "weight_decay": 1e-5}
        )
        # Parameters Matching Module
        param_groups.append(
            {"params": model.sequential[1].parameters(),
             "lr": 5e-4, "weight_decay": 1e-5}
        )
        if args.optimizer == "AdamW":
            optimizer = optim.AdamW(param_groups, lr=args.lr) #Changed to AdamW
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(param_groups, lr=args.lr)
        optimizers.append(optimizer)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    if args.use_checkpoint:
        stamp = args.use_checkpoint
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    solver = Solver(
        model=model, 
        args=args,
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizers,
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE
    )
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        # checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        solver.epoch, solver.best, solver.args = resume_if_possible(root, model, optimizers[0])
        # model.load_state_dict(checkpoint["model_state_dict"])
        # if args.use_two_optim:
        #     optimizers[0].load_state_dict(checkpoint["optimizer_3detr_state_dict"])
        #     optimizers[1].load_state_dict(checkpoint["optimizer_reference_state_dict"])
        # else:
        #     optimizers[0].load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)
    num_params = get_num_params(model)
    print(f"num_params_total is {get_num_params_total(model)}")
    return solver, num_params, root, args

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params["total"]
    info["num_params_detr"] = num_params["detr"]
    info["num_params_lang"] = num_params["lang"]
    info["num_params_matching"] = num_params["matching"]

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id                                        
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root, args = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #################################### [start] Debug arguments #######################################
    parser.add_argument("--compute_AP", default=False, action="store_true", help="Output AP metrics during the training to verify the performance of detector")


    #################################### [start] scanrefer arguments #######################################
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # FIXME: set the right batch_size
    parser.add_argument("--batch_size", required=True, type=int, help="batch size", default=12)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3) # 1e-3 is a better lr in the experiment so far
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", default=False, action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", default=False, action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", default=True, action="store_true", help="Use normal in input.")
    parser.add_argument("--use_height", default=True, action="store_true", help="Use height in input.")
    parser.add_argument("--use_multiview", default=True, action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--use_two_optim", action="store_true", help="Use 2 separate optimizers for detection and reference part.")

    #################################### [start] 3detr arguments #######################################
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "Adam"], help="Switch between AdamW and Adam.")
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=True, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )
    parser.add_argument(
        "--lang_type", default="bert", choices=["gru", "attention", "transformer_encoder", "bert"]
    )
    parser.add_argument(
        "--use_att_mask", action="store_true", default=True, help="Use the attention mask in the matching module."
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )

    ##### unfreeze #####
    parser.add_argument("--unfreeze_mlp_heads", action="store_true", default=True, help="Unfreeze the mlp detection heads.")
    parser.add_argument("--unfreeze_decoder_last_layeres", type=int, default=0, help="Unfreeze last few layers of the detection decoder.")

    ### Encoder
    parser.add_argument(
        "--enc_type", default="masked", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.5, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    # parser.add_argument("--use_color", default=True, action="store_true") comment out since scanrefer parser already have this arg field

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=1, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.25, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", default="scannet", type=str, choices=["scannet"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", required=True, default=6, type=int, help="number of workers for dataloader") # set to required
    # parser.add_argument("--batchsize_per_gpu", default=8, type=int) comment out since we can use "batchsize" arg field from scanrefer above

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    # parser.add_argument("--max_epoch", default=720, type=int) comment out since we can use "epoch" arg field from scanrefer above
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    # parser.add_argument("--seed", default=0, type=int) comment out since we can use "seed" arg field from scanrefer above

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)
    #################################### [end] 3detr arguments #######################################
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
    
