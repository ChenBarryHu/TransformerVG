import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule

sys.path.insert(1, "E:/Daten/Dokumente/GitHub/3dvg-transformer/_3detr")
#from _3detr.models.model_3detr import Model3DETR
#from _3detr.models import build_model
from _3detr.models import model_3detr as detr

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, args,
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=256):
        super().__init__()

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

        self.use_3detr = True

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        if not self.use_3detr:
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)

            # Vote aggregation and object proposal
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        else:
        # ------------- PROPOSAL GENERATION WITH 3DETR -------------

            pre_encoder = detr.build_preencoder(args)
            encoder = detr.build_encoder(args)
            decoder = detr.build_decoder(args)
            self.model_3detr = detr.Model3DETR(
                pre_encoder,
                encoder,
                decoder,
                num_classes=self.num_class,
                encoder_dim=args.enc_dim,
                decoder_dim=args.dec_dim,
                mlp_dropout=args.mlp_dropout,
                num_queries=args.nqueries,
            )
            output_processor = detr.BoxProcessor(self.num_class)

        # ----------------------------------------------------------

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)

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

        if not self.use_3detr:
        # --------- HOUGH VOTING ---------
            data_dict = self.backbone_net(data_dict)

            # --------- HOUGH VOTING ---------
            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features

            xyz, features = self.vgen(xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
            data_dict["vote_features"] = features

            # --------- PROPOSAL GENERATION ---------
            data_dict = self.proposal(xyz, features, data_dict)
        else:
            # --------- 3DETR ---------

            # TODO: Work in progress.
            #We have to change the output of the forward pass of 3DETR. ScanRefer keeps working with data_dict.
            data_dict = self.model_3detr(data_dict)

            #--------------------------

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict
