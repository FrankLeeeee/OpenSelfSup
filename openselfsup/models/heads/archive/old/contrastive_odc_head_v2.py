import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..utils import accuracy
from ..registry import HEADS


@HEADS.register_module
class ContrastiveODCHead_V2(nn.Module):
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
        super(ContrastiveODCHead_V2, self).__init__()
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

    def calc_feature_sim(self, feature_pairs):
        sim_pairs = [torch.matmul(f[0], f[1])
                     for f in feature_pairs]
        feature_sim = torch.stack(sim_pairs, dim=0)
        feature_sim = torch.unsqueeze(feature_sim, 1)

        # feature_sim = sum(sim_pairs) / len(sim_pairs)
        # feature_sim = torch.exp(feature_sim / self.temperature)
        # return feature_sim

        return feature_sim

    def _masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps/masked_sums)

    def calc_rel_dissim(self,
                        feature,
                        centroids):
        """Calculate relative contrastive loss

        Args:
            feature (Tensor): Input features of shape (N, D).
            centroids: centroids of the random_features (M, D)

        Returns:
            Tensor : relative contrastive loss
        """
        # calculate euclidean distant between centroids
        N = feature.size(0)
        mask = 1 - torch.eye(N, dtype=torch.uint8).cuda()

        cdist = torch.cdist(centroids,
                            centroids, p=2)  # N * N
        feature_sim = torch.matmul(
            feature, feature.permute(1, 0))

        feature_sim = torch.masked_select(
            feature_sim, mask == 1).reshape(N, -1)
        cdist = torch.masked_select(
            cdist, mask == 1).reshape(N, -1)

        # softmax_cdist = self._masked_softmax(cdist, cdist > 0, dim=1)

        # sum of dissimilarity across different clusters with distance-based weights

        # normalize distance to 1 - 10
        cdist = (1 - 0.1) * (cdist - cdist.min()) / \
            (cdist.max() - cdist.min()) + 0.1

        # print("cdist: {}".format(cdist))
        # print("feature_sim: {}".format(feature_sim))

        rel_dissim = feature_sim / cdist

        # print("rel_dissim: {}".format(rel_dissim))
        return rel_dissim

    def loss(self,
             feature_pairs,
             feature,
             cls_scores,
             cls_labels,
             centroids,
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
        feature_sim = self.calc_feature_sim(feature_pairs)
        rel_dissim = self.calc_rel_dissim(feature, centroids)
        N = feature.size(0)

        # print("feature_sim shape: {}".format(feature_sim.shape))
        # print("rel_dissim shape: {}".format(rel_dissim.shape))
        logits = torch.cat((feature_sim, rel_dissim), dim=1)
        # print("logits: {}".format(logits))

        # logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        rel_ctc_loss = self.contrastive_criterion(logits, labels)
        losses = dict()
        losses['cls_loss'] = cls_loss
        # losses['feature_sim_loss'] = feature_sim
        # losses['rel_dissim'] = rel_dissim
        losses['rel_ctc_loss'] = rel_ctc_loss
        losses['acc'] = acc
        # losses['loss'] = cls_loss + rel_ctc_loss
        losses['loss'] = cls_loss + rel_ctc_loss
        # print(losses)
        return losses
