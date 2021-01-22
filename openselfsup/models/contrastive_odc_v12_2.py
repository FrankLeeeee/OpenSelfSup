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
class ContrastiveODC_V12_2(nn.Module):
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
                 train=True,
                 ):
        super(ContrastiveODC_V12_2, self).__init__()
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
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

        self.rank, self.world_size = get_dist_info()

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
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

        # for contrastive loss
        z = feature[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        # for classification loss
        # choose the first feature to feed into memory bank
        feature_pairs = torch.split(
            feature[0], split_size_or_sections=2, dim=0)

        feature_to_odc = [pair[0]
                          for pair in feature_pairs]
        feature_to_cts = [pair[1]
                          for pair in feature_pairs]
        feature_to_odc = torch.stack(feature_to_odc, dim=0)
        feature_to_cts = torch.stack(feature_to_cts, dim=0)

        # projection head
        outs_to_odc = self.head([feature_to_odc])  # (2n) * k
        outs_to_cts = self.head([feature_to_cts])  # (2n) * k

        # get items for loss
        if self.memory_bank.label_bank.is_cuda:
            cls_labels = self.memory_bank.label_bank[idx]
        else:
            cls_labels = self.memory_bank.label_bank[idx.cpu()].cuda()

        # get centroids for the labels
        # centroids = self.memory_bank.centroids[cls_labels]

        # loss input
        loss_inputs = dict()
        loss_inputs['positive'] = positive
        loss_inputs['negative'] = negative
        loss_inputs['outs_to_odc'] = outs_to_odc
        loss_inputs['outs_to_cts'] = outs_to_cts
        loss_inputs['cls_scores'] = outs_to_odc
        loss_inputs['cls_labels'] = cls_labels
        # loss_inputs['centroids'] = centroids

        # loss calculation
        losses = self.head.loss(**loss_inputs)
        # print(losses)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature_to_odc.detach())
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
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            labels (numpy.ndarray): Label assignments. Default: None.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        if labels is None:
            if self.memory_bank.label_bank.is_cuda:
                labels = self.memory_bank.label_bank.cpu().numpy()
            else:
                labels = self.memory_bank.label_bank.numpy()
        hist = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5))**reweight_pow
        weight = inv_hist / inv_hist.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.head.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)
