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
from models.lang_module import LangModule
from models.match_module import MatchModule
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

        # FIXME-WINDOWS: set the weight_path to the correct path to 3detr pretrained weights
        # weight_path = "/home/shichenhu/3dvg-transformer/weights/scannet_ep1080_epoch_600/checkpoint_best.pth"
        weight_path = "/home/barry/dev/3dvg-3detr/outputs/experiment_6/checkpoint_best.pth"
        if os.path.isfile(weight_path):
            print("Loading pretrained 3detr weights")
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
        mlp_func_feature = partial(
            GenericMLP,
            use_conv=True,
            hidden_dims=[args.dec_dim,args.dec_dim],
            dropout=args.mlp_dropout,
            input_dim=args.dec_dim,
        )



        self.feature_head = mlp_func_feature(output_dim=128)


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        # self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        # self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        # self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        self.sequential = nn.ModuleList([self.feature_head])
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)


            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)
            self.sequential = nn.ModuleList([self.feature_head, self.lang, self.match])

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

        #scanrefer_features = self.feature_head(box_features).transpose(1, 2)
        scanrefer_features = self.sequential[0](box_features).transpose(1, 2)
        data_dict['scanrefer_features'] = scanrefer_features

        
        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            #data_dict = self.lang(data_dict)
            data_dict = self.sequential[1](data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            #data_dict = self.match(data_dict)
            data_dict = self.sequential[2](data_dict)

        return data_dict
