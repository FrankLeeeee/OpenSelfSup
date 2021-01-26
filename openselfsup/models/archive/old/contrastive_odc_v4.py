import numpy as np
import torch
import torch.nn as nn
import random

from openselfsup.utils import print_log
from mmcv.runner import get_dist_info
from . import builder
from .registry import MODELS
from .utils import Sobel, GatherLayer
import time
import functools
import multiprocessing
import math


def get_features(num, output, close_centroids, length, label_bank, feature_bank, feat_dim, i):
    mask = torch.where(
        label_bank == close_centroids[i][0], label_bank, torch.LongTensor([-1]))
    mask = mask.repeat(feat_dim).reshape(feat_dim, length).permute(1, 0)
    res = torch.masked_select(feature_bank, mask != -1).reshape(-1, feat_dim)

    if res.size(0) < num:
        times = math.ceil(num / res.size(0))
        res = res.repeat(times, 1)

    idx_keep = random.sample(range(res.size(0)), num)
    output[i] = feature_bank[idx_keep]


@ MODELS.register_module
class ContrastiveODC_V4(nn.Module):
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
                 train_bs=32,
                 neg_num=32):
        super(ContrastiveODC_V4, self).__init__()
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
        self.neg_num = neg_num

        if train:
            self.pool = multiprocessing.Pool(train_bs)
            # self.pool2 = multiprocessing.Pool(self.wor)

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
        return mask, pos_ind

    def get_close_cluster(self, num=2):
        centroids = self.memory_bank.centroids
        dis = torch.cdist(centroids, centroids, p=2)
        close_cluster = torch.zeros(
            (self.memory_bank.num_classes, num), dtype=torch.float32)

        idx = list(range(self.memory_bank.num_classes))
        for i in range(self.memory_bank.num_classes):
            idx_centroid_pairs = zip(dis[i], idx)
            idx_centroid_pairs = sorted(idx_centroid_pairs)[1:num+1]
            idx_keep = [item[1] for item in idx_centroid_pairs]
            close_cluster[i] = torch.FloatTensor(idx_keep)

        return close_cluster

    def get_close_features(self, labels, close_clusters, num=32):
        N = labels.size(0)
        close_centroids = close_clusters[labels]
        output = torch.zeros((labels.size(0), num, self.memory_bank.feat_dim))
        fn = functools.partial(get_features, num, output, close_centroids, self.memory_bank.length,
                               self.memory_bank.label_bank, self.memory_bank.feature_bank, self.memory_bank.feat_dim)

        self.pool.map(fn, range(N))
        return output

    def get_negative(self, features, labels, cluster_num=2, neg_num=16):
        N = features.size(0)

        label_list = [torch.zeros_like(labels) for i in range(self.world_size)]
        torch.distributed.all_gather(label_list, labels)
        labels = torch.cat(label_list, dim=0)

        # get features from feature banks
        # feature bank is only available on rank 0
        # randomly select neg_count feature samples from
        # 10% of the nearest features
        feature_collection = torch.zeros(
            (self.world_size * features.size(0), neg_num, features.size(1))).cuda()

        if self.rank == 0:
            close_clusters = self.get_close_cluster(cluster_num)
            feature_collection = self.get_close_features(
                labels, close_clusters, neg_num).cuda()

        torch.distributed.broadcast(feature_collection, src=0)
        negative = feature_collection[self.rank *
                                      features.size(0): (self.rank+1) * features.size(0)]
        return negative

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
        # forward & backward
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        img = img.reshape(
            img.size(0) * 2, img.size(2), img.size(3), img.size(4))  # 2n

        x = self.forward_backbone(img)  # 2n
        feature = self.neck(x)  # (2n) * d

        # check for nan
        # for i in range(feature[0].size(0)):
        #     if sum(torch.isnan(feature[0][i])):
        #         print("img: {}".format(img))
        #         print("x: {}".format(x))
        #         print("feature: {}".format(feature))
        #         raise Exception("feature has nan error: {}".format(i))

        # for contrastive loss
        z = feature[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)

        # for classification loss
        # choose the first feature to feed into memory bank
        feature_pairs = torch.split(
            z, split_size_or_sections=2, dim=0)

        feature = [pair[0]
                   for pair in feature_pairs]
        feature = torch.stack(feature, dim=0)

        # projection head
        outs = self.head([feature])  # (2n) * k

        # get items for loss
        if self.memory_bank.label_bank.is_cuda:
            cls_labels = self.memory_bank.label_bank[idx]
        else:
            cls_labels = self.memory_bank.label_bank[idx.cpu()].cuda()

        # contrastive loss
        # torch.cuda.synchronize()
        # cts_start = time.time()
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        # get positive
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1

        # select negative, (2N)x(32)
        # So far only 1 cluster is allowed
        # torch.cuda.synchronize()
        # get_neg_start = time.time()

        negative_feature = self.get_negative(
            feature, cls_labels, 1, self.neg_num)

        # torch.cuda.synchronize()
        # get_neg_end = time.time()
        # print("get neg: {}".format(get_neg_end - get_neg_start))

        negative = torch.zeros((z.size(0), self.neg_num)).cuda()

        for i in range(N):
            negative[i*2] = torch.matmul(z[i*2],
                                         negative_feature[i].permute(1, 0))
            negative[i*2+1] = torch.matmul(z[i*2+1],
                                           negative_feature[i].permute(1, 0))
        # torch.cuda.synchronize()
        # cts_end = time.time()

        # print("for negative output: {}".format(cts_end - get_neg_end))
        # print("for contrastive part: {}".format(cts_end - cts_start))

        # get centroids for the labels
        # centroids = self.memory_bank.centroids[cls_labels]

        # loss input
        loss_inputs = dict()
        loss_inputs['positive'] = positive
        loss_inputs['negative'] = negative
        loss_inputs['cls_scores'] = outs
        loss_inputs['cls_labels'] = cls_labels
        # loss_inputs['centroids'] = centroids

        # loss calculation
        losses = self.head.loss(**loss_inputs)
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
