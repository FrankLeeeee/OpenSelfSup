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
class ContrastiveODC_V25(nn.Module):
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
                 num_neg_centroids=16,
                 num_neg_features=128,
                 ):
        super(ContrastiveODC_V25, self).__init__()
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
        self.num_classes = head.num_classes
        self.num_neg_centroids = num_neg_centroids
        self.num_neg_features = num_neg_features

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

    def get_features(self, labels):
        # sample pos and neg features
        pos_feature_idx, neg_feature_idx = self._get_feature_idx(labels)
        pos_feature_idx = torch.LongTensor(pos_feature_idx).cuda().flatten()
        neg_feature_idx = torch.LongTensor(neg_feature_idx).cuda().flatten()

        # create tensor list for all gather
        pos_feature_idx_list = [torch.zeros_like(pos_feature_idx) for i in range(self.world_size)]
        neg_feature_idx_list = [torch.zeros_like(neg_feature_idx) for i in range(self.world_size)]

        # gather the pos and neg feat idx
        torch.distributed.all_gather(pos_feature_idx_list, pos_feature_idx)
        torch.distributed.all_gather(neg_feature_idx_list, neg_feature_idx)

        # format the pos and neg feat idx
        pos_feature_idx_list = torch.cat(pos_feature_idx_list, dim=0)
        neg_feature_idx_list = torch.cat(neg_feature_idx_list, dim=0)

        # preprare feature collections for broadcast
        pos_feature_collection = torch.zeros(
            size=(labels.size(0) * self.world_size, self.memory_bank.feat_dim)
        ).cuda()
        neg_feature_collection = torch.zeros(
            size=(labels.size(0) * self.world_size * self.num_neg_features, 
            self.memory_bank.feat_dim)
        ).cuda()

        # get the features
        if self.rank == 0:
            pos_feature_collection = self.memory_bank.feature_bank[pos_feature_idx_list][:].cuda()
            neg_feature_collection = self.memory_bank.feature_bank[neg_feature_idx_list][:].cuda()
        torch.distributed.broadcast(pos_feature_collection, src=0)
        torch.distributed.broadcast(neg_feature_collection, src=0)

        # get the features of the current rank
        bs = labels.size(0)
        pos_feature = pos_feature_collection[self.rank*bs:(self.rank+1)*bs]
        neg_feature = neg_feature_collection[self.rank*bs*self.num_neg_features:(self.rank+1)*bs*self.num_neg_features]
        neg_feature = neg_feature.view(bs, self.num_neg_features, -1)
        return pos_feature, neg_feature


    def _get_feature_idx(self, labels):
        # calcuate close cluster
        close_clsuter = self._get_close_cluster()

        # prepare index list
        pos_idx_list = []
        neg_idx_list = []

        # get idx
        for label in labels:
            # sample pos feature idx
            pos_feature_idx = (self.memory_bank.label_bank == label).nonzero(as_tuple=False).flatten().tolist()
            pos_feature_idx = random.sample(pos_feature_idx, 1)
            pos_idx_list.extend(pos_feature_idx)

            # sample neg feature idx
            close_cluster_idx_list = close_clsuter[label]
            neg_feature_idx_all = []

            for cluster_idx in close_cluster_idx_list:
                neg_feature_idx = (self.memory_bank.label_bank == cluster_idx).nonzero(as_tuple=False).flatten().tolist()
                neg_feature_idx_all.extend(neg_feature_idx)
            neg_feature_idx_all = random.sample(neg_feature_idx_all, self.num_neg_features)
            neg_idx_list.append(neg_feature_idx_all)
        
        return pos_idx_list, neg_idx_list


    def _get_close_cluster(self):
        dis = torch.cdist(self.memory_bank.centroids, self.memory_bank.centroids)
        val, idx = dis.sort(dim=1)
        idx = idx[:, 1:self.num_neg_centroids+1]
        return idx

    def get_updated_features(self, idx):
        # create tensor list for all gather
        bs = idx.size(0)
        idx_list = [torch.zeros_like(idx) for i in range(self.world_size)]
        torch.distributed.all_gather(idx_list, idx)
        idx_list = torch.cat(idx_list, dim=0)
        idx_collection = torch.zeros(
            size=(bs*self.world_size, self.memory_bank.feat_dim)
        ).cuda()

        if self.rank == 0:
            idx_collection = self.memory_bank.feature_bank[idx_list][:].cuda()
        
        torch.distributed.broadcast(idx_collection, src=0)
        idx_features = idx_collection[self.rank*bs: (self.rank+1)*bs]

        return idx_features


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
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        img = img.reshape(
            img.size(0) * 2, img.size(2), img.size(3), img.size(4))  # 2n

        x = self.forward_backbone(img)  # 2n
        feature = self.neck(x)  # (2n) * d

        # for instance-level contrastive loss
        z = feature[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        ins_pos = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        ins_neg = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        
        # get labels
        if self.memory_bank.label_bank.is_cuda:
            cls_labels = self.memory_bank.label_bank[idx]
        else:
            cls_labels = self.memory_bank.label_bank[idx.cpu()].cuda()

        # choose the first feature to feed into memory bank
        feature_pairs = torch.split(
            feature[0], split_size_or_sections=2, dim=0)
        feature_to_odc = [pair[0]
                   for pair in feature_pairs]
        feature_to_cts = [pair[1]
                   for pair in feature_pairs]
        feature_to_odc = torch.stack(feature_to_odc, dim=0)
        feature_to_cts = torch.stack(feature_to_cts, dim=0)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature_to_odc.detach())


        # for cluster level contrastive loss
        # cluster level contrastive loss
        feat_after_update = self.get_updated_features(idx)
        pos_features, neg_features = self.get_features(cls_labels)

        # calculate simlilarity for contrastive loss
        
        cls_pos = torch.mul(feat_after_update, pos_features).sum(dim=1).unsqueeze(1)
        cls_neg = torch.mul(feat_after_update.unsqueeze(1), neg_features).sum(dim=2)
        

        # loss input
        loss_inputs = dict()
        loss_inputs['instance_positive'] = ins_pos
        loss_inputs['instance_negative'] = ins_neg
        loss_inputs['cluster_positive'] = cls_pos
        loss_inputs['cluster_negative'] = cls_neg
        loss_inputs['feature_to_odc'] = feature_to_odc
        loss_inputs['feature_to_cts'] = feature_to_cts
        
        # loss calculation
        losses = self.head.loss(**loss_inputs)

        
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
