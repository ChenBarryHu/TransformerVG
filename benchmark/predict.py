import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from scripts.visualize import write_bbox, write_ply_rgb, align_mesh
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.refnet import RefNet
from utils.box_util import get_3d_box
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANNET_ROOT = "/home/barry/dev/ScanRefer/data/scannet/scans_test" # TODO point this to your scannet data
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id 
SCANREFER_TEST = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_test.json")))

def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=args.use_height,
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview
    )
    print("predict for {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataset_num_workers)

    return dataset, dataloader

def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    model = RefNet(
        args=args,
        dataset_config=config,
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir
    ).cuda()

    model_name = "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=True)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):
    scanrefer = SCANREFER_TEST
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def predict(args):
    print("predict bounding boxes...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "test", DC)

    # model
    model = get_model(args, DC)
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred")
    os.makedirs(dump_dir, exist_ok=True)
    dumped_scene_id = []

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # predict
    print("predicting...")
    pred_bboxes = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        # feed
        data_dict = model(data_dict)
        # we do not calculate loss here, since there is no ground truth for reference in the test set
        # _, data_dict = get_loss(
        #     data_dict=data_dict, 
        #     args=args,
        #     config=DC, 
        #     detection=False,
        #     reference=True
        # )

        objectness_preds_batch = (data_dict['objectness_prob'] > 0.5).long()

        if POST_DICT:
            _ = parse_predictions(data_dict, POST_DICT)
            nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        point_clouds = data_dict['point_clouds'].cpu().numpy()
        pcl_color = data_dict['pcl_color'].cpu().numpy()
        pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1).cpu().numpy() # (B,)
        pred_center = data_dict['center_unnormalized'].cpu().numpy() # (B,K,3)
        pred_size = data_dict['size_unnormalized'].cpu().numpy() # (B,K,3)
        pred_angle = data_dict['angle_continuous'].cpu().numpy()

        # the following data fields are unnecessary for calculating bounding boxes
        # pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        # pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        # pred_heading_class = pred_heading_class # B,num_proposal
        # pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        # pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        # pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        # pred_size_class = pred_size_class
        # pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            # DC.param2obb is not used here, since we can directly calculate the pred_obb from 
            # the predicted size and angle 
            # pred_obb = DC.param2obb(
            #     pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
            #     pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
            #     pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
            #     pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
            #     pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
            # )
            pred_obb = np.zeros((7,))
            pred_obb[0:3] = pred_center[i, pred_ref_idx]
            pred_obb[3:6] = pred_size[i, pred_ref_idx]
            pred_obb[6] = pred_angle[i, pred_ref_idx]
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

            # construct the multiple mask
            multiple = data_dict["unique_multiple"][i].item()

            # construct the others mask
            others = 1 if data_dict["object_cat"][i] == 17 else 0

            # store data
            scanrefer_idx = data_dict["scan_idx"][i].item()
            pred_data = {
                "scene_id": scanrefer[scanrefer_idx]["scene_id"],
                "object_id": scanrefer[scanrefer_idx]["object_id"],
                "ann_id": scanrefer[scanrefer_idx]["ann_id"],
                "bbox": pred_bbox.tolist(),
                "unique_multiple": multiple,
                "others": others
            }
            pred_bboxes.append(pred_data)

            # visualize the mesh, point clood and bounding boxes
            scene_id = pred_data["scene_id"]
            scene_dump_dir = os.path.join(dump_dir, scene_id)
            if scene_id not in dumped_scene_id:
                os.makedirs(scene_dump_dir, exist_ok=True)
                mesh = align_mesh(scene_id)
                print(f"mesh output to {os.path.join(scene_dump_dir, 'mesh.ply')}")
                mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))
                dumped_scene_id.append(scene_id)
                write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))

            write_bbox(pred_obb, 0, os.path.join(scene_dump_dir, 'pred_{}_{}_{}.ply'.format(pred_data["object_id"], scanrefer[scanrefer_idx]["object_name"],pred_data["ann_id"])))

    # dump
    print("dumping...")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.json")
    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_height", default=True, action="store_true", help="Use height in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")


    # other arguments from TrasnformerVG
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument(
        "--lang_type", default="gru", choices=["gru", "attention", "transformer_encoder", "bert"]
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
    ### Encoder
    parser.add_argument(
        "--enc_type", default="masked", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.3, type=float)
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

    parser.add_argument("--dataset_num_workers", required=True, default=4, type=int) # set to required, works well with 6 on 3080ti
    # parser.add_argument("--batchsize_per_gpu", default=8, type=int) comment out since we can use "batchsize" arg field from scanrefer above


    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    predict(args)
