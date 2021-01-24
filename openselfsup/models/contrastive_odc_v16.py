import functools
import math
import multiprocessing
import random
import time

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import get_dist_info
from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import GatherLayer, Sobel


@ MODELS.register_module
class ContrastiveODC_V16(nn.Module):
    """Contrastive Learning with ODC.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        with_sobel (bool): Whether to apply a Sobel filter on images. Default: False.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        memory_bank (dict): Module of memory banks. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 neck=None,
                 head=None,
                 memory_bank=None,
                 pretrained=None,
                 ):
        super(ContrastiveODC_V16, self).__init__()
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        if head is not None:
            self.head = builder.build_head(head)
        if memory_bank is not None:
            self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)

        # set reweight tensors
        # self.num_classes = head.num_classes
        # self.loss_weight = torch.ones((self.num_classes, ),
        #                               dtype=torch.float32).cuda()
        # self.loss_weight /= self.loss_weight.sum()

        self.rank, self.world_size = get_dist_info()

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(
                pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')
        self.head.init_weights(init_linear='normal')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    @ staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
            1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def get_features(self, idx, neg_num):
        # gather feature index
        bs = idx.size(0)
        pos_idx_list = [torch.zeros_like(idx) for i in range(self.world_size)]
        torch.distributed.all_gather(pos_idx_list, idx)
        pos_idx_list = torch.cat(pos_idx_list, dim=0)

        # gather neg feature index
        neg_indices = torch.randint(0, self.memory_bank.length, size=(bs, neg_num*2)).cuda()
        neg_indices = [neg_indices[i][(neg_indices[i] - idx[i]).nonzero()][:neg_num] for i in range(bs)]
        neg_indices = torch.stack(neg_indices, dim=0)
        neg_indices_list = [torch.zeros_like(neg_indices) for i in range(self.world_size)]
        torch.distributed.all_gather(neg_indices_list, neg_indices)
        neg_indices_list = torch.cat(neg_indices_list, dim=0).flatten()
        

        # create feature tensor
        feat_dim = self.memory_bank.feat_dim
        old_feature_collection = torch.zeros(
            (self.world_size * bs, feat_dim)).cuda()
        other_cluster_feature = torch.zeros(
            (self.world_size * bs * neg_num), feat_dim
        ).cuda()

        if self.rank == 0:
            old_feature_collection = self.memory_bank.feature_bank[pos_idx_list][:].cuda()
            other_cluster_feature = self.memory_bank.feature_bank[neg_indices_list][:].cuda()
            
        torch.distributed.broadcast(old_feature_collection, src=0)
        torch.distributed.broadcast(other_cluster_feature, src=0)

        old_features = old_feature_collection[self.rank * bs: (self.rank+1) * bs]
        other_cluster_feature = other_cluster_feature[self.rank * bs * neg_num: (self.rank+1) * bs * neg_num].view(bs, neg_num, -1)
        return old_features, other_cluster_feature
    
    def get_close_cluster(self, num=2):
        centroids = self.memory_bank.centroids
        dis = torch.cdist(centroids, centroids, p=2)
        val, idx = dis.sort()
        close_cluster = idx[:, 1:num+1].cpu()

        return close_cluster

    def forward_train(self, img, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.forward_backbone(img)
        feature = self.neck(x)[0]

        # instance contrastive
        ins_pos, ins_neg = self.get_features(idx, 32)
        ins_pos_sim = torch.mul(feature, ins_pos).sum(dim=1).unsqueeze(1)
        ins_neg_sim = torch.mul(feature.unsqueeze(1), ins_neg).sum(dim=2)


        # cluster contrastive
        if self.memory_bank.label_bank.is_cuda:
            cls_labels = self.memory_bank.label_bank[idx]
        else:
            cls_labels = self.memory_bank.label_bank[idx.cpu()].cuda()
        
        centroids = self.memory_bank.centroids[cls_labels]
        close_centroids = self.get_close_cluster(num=16)
        close_centroids = close_centroids[cls_labels]

        neg_centroids = [
            self.memory_bank.centroids[close_centroids[i]]
            for i in range(close_centroids.size(0))
        ]
        neg_centroids = torch.stack(neg_centroids, dim=0)

        cluster_pos_sim = torch.mul(feature, centroids).sum(dim=1).unsqueeze(1)
        cluster_neg_sim = torch.mul(feature.unsqueeze(1), neg_centroids).sum(dim=2)

        
        # loss input
        loss_inputs = dict()
        loss_inputs['instance_positive'] = ins_pos_sim
        loss_inputs['instance_negative'] = ins_neg_sim
        loss_inputs['cluster_positive'] = cluster_pos_sim
        loss_inputs['cluster_negative'] = cluster_neg_sim

        # loss calculation
        losses = self.head(**loss_inputs)
        # print(losses)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature.detach())
        losses['change_ratio'] = change_ratio

        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))

    def set_reweight(self, labels=None, reweight_pow=0.5):
        pass
        # """Loss re-weighting.

        # Re-weighting the loss according to the number of samples in each class.

        # Args:
        #     labels (numpy.ndarray): Label assignments. Default: None.
        #     reweight_pow (float): The power of re-weighting. Default: 0.5.
        # """
        # if labels is None:
        #     if self.memory_bank.label_bank.is_cuda:
        #         labels = self.memory_bank.label_bank.cpu().numpy()
        #     else:
        #         labels = self.memory_bank.label_bank.numpy()
        # hist = np.bincount(
        #     labels, minlength=self.num_classes).astype(np.float32)
        # inv_hist = (1. / (hist + 1e-5))**reweight_pow
        # weight = inv_hist / inv_hist.sum()
        # self.loss_weight.copy_(torch.from_numpy(weight))
        # self.head.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)
