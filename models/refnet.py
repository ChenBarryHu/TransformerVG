import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.join( os.path.join( os.getcwd(), os.pardir ),"detr"))
sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from functools import partial
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule, LangModuleAttention, LangModuleBert, LangModuleTransEncoder
from models.match_module import MatchModule
from models._3dvg_match_module import MatchModule as dvg_matchmodule
from _3detr.models import build_model
from _3detr.datasets import build_dataset
from _3detr.models.helpers import GenericMLP



class RefNet(nn.Module):
    def __init__(self, args, dataset_config, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=256):
        super().__init__()
        self.detr, _ = build_model(args, dataset_config)

        # FIXME: set the weight_path to the correct path to 3detr-m (masked) pretrained weights
        # weight_path = "E:/Daten/ADL4CV/Pretrained_3detr/experiment_14/checkpoint_best.pth"
        weight_path = "/home/barry/dev/3dvg-3detr/outputs/experiment_14/checkpoint_best.pth"

        if os.path.isfile(weight_path):
            print(f"Loading pretrained 3detr weights from {weight_path}")
            weights = torch.load(weight_path)
            self.detr.load_state_dict(weights['model'])
        else:
            print("Using untrained RefNet")

        # freeze the weights of detector so we can focus on other modules
        freeze = True
        if freeze:
            for param in self.detr.parameters():
                param.requires_grad = False
        


        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference
        self.use_att_mask = args.use_att_mask
        mlp_func_feature = partial(
            GenericMLP,
            use_conv=True,
            hidden_dims=[args.dec_dim,args.dec_dim],
            dropout=args.mlp_dropout,
            input_dim=args.dec_dim,
        )




        # self.feature_head = mlp_func_feature(output_dim=128)


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        # self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        # self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        # self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)


        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            # to compare strings, use "==" instead of "is" 
            # The "==" operator compares the value or equality of two objects, 
            # whereas the Python "is" operator checks whether two variables point to the same object in memory.
            if args.lang_type == "gru":

                lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size)

            elif args.lang_type == "attention":
                lang = LangModuleAttention(
                    num_class, 
                    use_lang_classifier,
                    embed_dim=300,
                    num_head=4,
                    dropout=0.1,
                    batch_first=True
                )
            elif args.lang_type == "transformer_encoder":
                lang = LangModuleTransEncoder(
                    num_class, 
                    use_lang_classifier,
                    embed_dim=300,
                    num_head=4,
                    dropout=0.1,
                    batch_first=True
                )
            elif args.lang_type == "bert":
                lang = LangModuleBert(
                    num_class, 
                    use_lang_classifier,
                    embed_dim=768,
                    num_head=4,
                    dropout=0.1,
                    batch_first=True
                )

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            use_3dvg = True
            if use_3dvg:
                match = dvg_matchmodule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size, use_att_mask=self.use_att_mask)
            else:
                match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)
            self.sequential = nn.ModuleList([lang, match]) # self.feature_head removed


    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        self.detr(data_dict)
        box_features = data_dict["box_features"]

        # scanrefer_features = self.feature_head(box_features).transpose(1, 2)
        # scanrefer_features = self.sequential[0](box_features).transpose(1, 2)
        # data_dict['scanrefer_features'] = box_features.transpose(1, 2)

        
        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------

            # self.lang(data_dict)
            data_dict = self.sequential[0](data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            #data_dict = self.match(data_dict)
            data_dict = self.sequential[1](data_dict)

        return data_dict
