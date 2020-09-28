import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS


@HEADS.register_module
class ContrastiveODCHead(nn.Module):
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
        super(ContrastiveODCHead, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

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

    def calc_cls_loss(self, cls_scores, cls_labels):
        assert isinstance(cls_scores, (tuple, list)) and len(cls_scores) == 1
        cls_loss = self.criterion(cls_score[0], cls_labels)
        acc = accuracy(cls_score[0], cls_labels)
        return cls_loss, acc

    def calc_feature_sim(self, features):
        feature_pairs = torch.split(
            features[0], split_size_or_sections=2, dim=0)
        feature_sim = sum([torch.matmul(f1, f2) for f1, f2 in feature_pairs])
        return -feature_sim

    def _masked_softmax(self, vec, mask, dim=1, epsilon=1e-5):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps/masked_sums)

    def calc_rel_cts_loss(self,
                          new_features,
                          old_features,
                          new_features_centroids,
                          random_features,
                          random_centroids):
        """Calculate relative contrastive loss

        Args:
            new_features (Tensor): Input features of shape (N, D).
            old_features (Tensor): the previous feature sample in 
                the memory with the same index of each feature (N, D).
            new_features_centroids: the centroids for the new features (N, D)
            random_features: some random features from different centroids (M, D)
            random_centroids: centroids of the random_features (M, D)

        Returns:
            Tensor : relative contrastive loss
        """
        # calculate euclidean distant between centroids
        cdist = torch.cdist(new_features_centroids,
                            random_centroids, p=2)  # N * N
        softmax_cdist = self._masked_softmax(cdist, cdist > 0, dim=1)
        feature_sim = torch.matmul(
            new_features, random_features.permute(1, 0))  # N * N

        # sum of dissimilarity across different clusters with distance-based weights
        feature_dissim = (softmax_cdist * feature_sim).sum()

        # sum of dissimilarity between images of the same cluster
        within_cluster = softmax_cdist.apply_(lambda x: 0 if x > 0 else 1)
        within_cluster_dissim = (within_cluster * feature_dissim).sum()

        # calculate similarity between new and old features
        new_old_feature_sim = (new_features * old_features).sum(dim=1)

        # final relative contrastive loss
        numerator = torch.exp(new_old_feature_sim)
        denominator = self.alpha * \
            torch.exp(within_cluster_dissim) + \
            self.beta * torch.exp(feature_dissim)

        rel_cts_loss = -torch.log(numerator/denominator)
        return rel_cts_loss

    def loss(self,
             new_features_pairs,
             new_features_mean,
             old_features,
             new_prjections,
             cls_labels,
             new_features_centroids,
             random_features,
             random_centroids):
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

        cls_loss = self.calc_cls_loss(new_prjections, cls_labels)
        feature_sim = self.calc_feature_sim(new_features_pairs)
        rel_ctc_loss = self.calc_rel_cts_loss(
            new_features_mean, old_features, new_features_centroids, random_features, random_centroids)

        losses['loss'] = cls_loss + feature_sim + rel_ctc_loss
        return losses
