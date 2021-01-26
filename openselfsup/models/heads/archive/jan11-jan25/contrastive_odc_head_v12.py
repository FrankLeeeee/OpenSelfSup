import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..registry import HEADS
from ..utils import accuracy


@HEADS.register_module
class ContrastiveODCHead_V12(nn.Module):
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
        super(ContrastiveODCHead_V12, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.temperature = temperature

        # for classification
        self.criterion = nn.CrossEntropyLoss()

        # for contrastive loss
        self.contrastive_criterion = nn.CrossEntropyLoss()
        self.cls_score_contrastive_criterion = nn.MSELoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

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
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def calc_cls_loss(self, cls_score, cls_labels):
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        cls_loss = self.criterion(cls_score[0], cls_labels)
        acc = accuracy(cls_score[0], cls_labels)
        return cls_loss, acc

    def calc_cts_loss(self, pos, neg):
        """
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        """

        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.contrastive_criterion(logits, labels)
        return loss

    def calc_cls_score_ctc_loss(self, outs_to_odc, outs_to_cts):
        return self.cls_score_contrastive_criterion(outs_to_odc[0], outs_to_cts[0])

    def loss(self,
             positive,
             negative,
             outs_to_odc,
             outs_to_cts,
             cls_scores,
             cls_labels,
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

        cls_loss, acc = self.calc_cls_loss(cls_scores, cls_labels)
        cts_loss = self.calc_cts_loss(positive, negative)
        cls_cts_loss = self.calc_cls_score_ctc_loss(
            outs_to_odc, outs_to_cts)
        losses = dict()
        losses['cls_loss'] = cls_loss
        losses['cts_loss'] = cts_loss
        losses['cls_cts_loss'] = cls_cts_loss
        losses['acc'] = acc
        losses['loss'] = cls_loss + cts_loss + cls_cts_loss
        # losses['loss'] = cls_loss
        # print(losses)
        return losses
