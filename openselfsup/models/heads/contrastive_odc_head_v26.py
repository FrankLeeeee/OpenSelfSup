import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..registry import HEADS
from ..utils import accuracy


@HEADS.register_module
class ContrastiveODCHead_V26(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
                 alpha=0.2,
                 beta=1,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 temperature=0.1):
        super(ContrastiveODCHead_V26, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.temperature = temperature

        # for classification
        self.criterion = nn.CrossEntropyLoss()
        self.instance_ctc_criterion = nn.CrossEntropyLoss()
        self.classification_score_criterion = nn.MSELoss()
        self.cluster_ctc_criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pass

    def calc_classification_loss(self, cls_score, cls_labels):
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        cls_loss = self.criterion(cls_score[0], cls_labels)
        acc = accuracy(cls_score[0], cls_labels)
        return cls_loss, acc

    def calc_ins_cts_loss(self, pos, neg):
        """
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        """

        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.instance_ctc_criterion(logits, labels)
        return loss

    def calc_cls_cts_loss(self, pos, neg):
        """
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        """

        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.cluster_ctc_criterion(logits, labels)
        return loss

    def calc_classification_score_loss(self, outs_to_odc, outs_to_cts):
        return self.classification_score_criterion(outs_to_odc, outs_to_cts)


    def loss(self,
             instance_positive,
             instance_negative,
             cluster_positive,
             cluster_negative,
             outs_to_odc,
             outs_to_cts,
             cls_scores,
             cls_labels
             ):
        """Forward head.

        Args:
            new_features_pairs (Tensor): (2N) x D output of the neck.
            new_features_mean (Tensor): N x D mean of feature pairs.
            old_features (Tensor): N x D previous features of the same index.
            new_prjections (Tensor): the projections of the new_features_mean
            cls_labels (Tensor): N x 1 cluster label of the new_features_mean.
            new_features_centroids (Tensor): N x D the centroids of the clusters.
            random_features (Tensor): M x D random features of other images.
            random_centroids (Tensor): N x D centroids of random_features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ins_cts_loss = self.calc_ins_cts_loss(instance_positive, instance_negative)
        cls_cts_loss = self.calc_cls_cts_loss(cluster_positive, cluster_negative)
        cls_score_mse_loss = self.calc_classification_score_loss(outs_to_odc, outs_to_cts)
        cls_loss, acc = self.calc_classification_loss(cls_scores, cls_labels)
        
        losses = dict()
        losses['ins_cts_loss'] = ins_cts_loss
        losses['cls_cts_loss'] = cls_cts_loss
        losses['cls_loss'] = cls_loss
        losses['acc'] = acc
        losses['cls_score_mse_loss'] = cls_score_mse_loss
        # losses['loss'] = 0.3 * ins_cts_loss  + 0.3 * cls_score_mse_loss + 0.3 * cls_loss  + 0.1 * cls_cts_loss
        losses['loss'] = ins_cts_loss + cls_cts_loss + cls_score_mse_loss + cls_loss
        
        return losses
