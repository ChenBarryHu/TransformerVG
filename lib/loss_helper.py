# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from _3detr.criterion import build_criterion
from _3detr.main import make_args_parser
import _3detr.datasets.scannet as detr_scannet
from _3detr.utils.dist import (
    all_reduce_average,
    reduce_dict,
)


FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness



def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['center_unnormalized']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    # objectness_scores = data_dict['objectness_scores']
    # criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    # objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    # objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
    

    return objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_reference_loss(data_dict, config):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    cluster_preds = data_dict["cluster_ref"] # (B*lang_num_max, num_proposal)



    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape #B*lang_num_max
    labels = np.zeros((batch_size, num_proposals))

    box_corners = data_dict["box_corners"][:, None, :, :, :].repeat(1, data_dict["lang_len_list"].shape[1], 1, 1,
                                                                    1).view(batch_size, num_proposals, 8, 3)
    box_corners_3detr = box_corners.detach().cpu().numpy()

    dim_1_gt_box_corners = data_dict["gt_box_corners"].shape[1]
    gt_box_corners = data_dict['gt_box_corners'][:, None, :, :, :].repeat(1,
                                                                          data_dict["lang_len_list"].shape[1], 1, 1, 1)
    gt_box_corners = gt_box_corners.view(batch_size, dim_1_gt_box_corners, 8, 3)

    for i in range(batch_size):
        ref_idx = data_dict["ref_box_label_list"].reshape(batch_size, -1)[i].argmax().item()
        gt_bbox_batch_3detr = gt_box_corners[i][ref_idx].cpu().numpy()

        # we calcualte iou using the 3detr output corners and gt_corners
        ious = box3d_iou_batch(box_corners_3detr[i], np.tile(gt_bbox_batch_3detr, (num_proposals, 1, 1)))
        # ious = box3d_iou_batch(box_corners_3detr[i], np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        # ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))

        # DEBUG the following two lines are for debug use, for checking how well predicted boxes overlaps with reference ground truth
        # obj_cat = data_dict["object_cat"][i]
        # print(f"max_iou with reference box: {ious.max()}, gt_reference_label: {config.class2type[obj_cat.item()]}")
        
        labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss

    bs = data_dict["lang_num"].shape[0]
    cluster_labels = cluster_labels.view(bs, -1, num_proposals)  # bs x lang_num_max x num_proposals
    cluster_preds = cluster_preds.view(bs, -1, num_proposals)  # bs x lang_num_max x num_proposals
    criterion = SoftmaxRankingLoss()
    loss = 0
    for i in range(bs):
        lang_num = data_dict["lang_num"][i]
        loss += criterion(cluster_preds[i, :lang_num, :], cluster_labels[i, :lang_num, :].float().clone())
    loss = loss / bs
    #loss = criterion(cluster_preds, cluster_labels.float().clone())
    lang_num_max = cluster_labels.shape[1]
    cluster_labels = cluster_labels.view(bs*lang_num_max, num_proposals)  # bs x lang_num_max x num_proposals
    cluster_preds = cluster_preds.view(bs*lang_num_max, num_proposals)

    return loss, cluster_preds, cluster_labels

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    # We can only calculate the loss from actual description and not the copies of these descriptions.
    # Check how many actual descriptions we have per sample.
    batch_size = data_dict["lang_scores"].shape[0]
    loss = 0
    for i in range(batch_size):
        num_descriptions = data_dict["lang_num"][i]
        loss += criterion(data_dict["lang_scores"][i, :num_descriptions, :],
                          data_dict["object_cat_list"][i, :num_descriptions])
    loss = loss/batch_size
    #Bxlang_num_maxxnum_classes and Bxlang_num_max
    return loss

def get_loss(data_dict, args, config, detection=True, reference=True, use_lang_classifier=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # TODO: experiment whether we should use 3detr's assignment method
    # Run objectness loss from scanrefer, we reuse these label, mask and assignment from scanrefer
    objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]

    # update the data_dict
    data_dict['objectness_label'] = objectness_label
    data_dict['objectness_mask'] = objectness_mask
    data_dict['object_assignment'] = object_assignment
    data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']


    # vote_loss = compute_vote_loss(data_dict) # voteloss should be zero, will remove it later

    # 3detr_loss
    detr_criterion = build_criterion(args, config)
    detr_criterion = detr_criterion.cuda(0)
    # detr_loss = detr_criterion()

    # Obj loss, Box loss and sem cls loss for 3detr:
    loss, loss_dict = detr_criterion(data_dict["boxes_prediction_3detr"], data_dict)

    # print(f"loss dimension: {loss.shape}")
    # print(f"loss_dict: {loss_dict}")
    loss_reduced = all_reduce_average(loss)
    loss_dict_reduced = reduce_dict(loss_dict)
    data_dict['3detr_loss'] = loss

    if detection:

        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        # data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()
    else:
        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        # data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if reference:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    else:
        # # Reference loss
        # ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        # data_dict["cluster_labels"] = cluster_labels
        
        #TODO: I don't think this will work with the new dataset loading.
        data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda() #This is needed in eval.
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()  #This is needed in eval.

        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()

    if reference and use_lang_classifier:
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()

    # Final loss function
    # FIXME: Choose the right loss function

    # the old scanrefer loss
    # loss = data_dict['vote_loss'] + 0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] \
    #     + 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]

    # to only train (ref, lang) uncomment the next line
    # loss = 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]
    
    # to tune everything (detection, ref, lang) uncomment the next line
    loss = data_dict['3detr_loss'] + 0.3*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]
    
    loss *= 10 # amplify
    data_dict['loss'] = loss

    return loss, data_dict
