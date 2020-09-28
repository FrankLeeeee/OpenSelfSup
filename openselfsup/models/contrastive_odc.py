import numpy as np
import torch
import torch.nn as nn
import random

from openselfsup.utils import print_log
from . import builder
from .registry import MODELS
from .utils import Sobel


@MODELS.register_module
class ConODC(nn.Module):
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
                 sample_size=50):
        super(ODC, self).__init__()
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
        self.sample_size = sample_size

        # set reweight tensors
        self.num_classes = head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

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

        # average the outs to feed into memory bank
        feature_pairs = torch.split(
            feature[0], split_size_or_sections=2, dim=0)
        mean_feature = [pair.mean(dim=0, keepdim=True)
                        for pair in feature_pairs]
        mean_feature = torch.concat(mean_feature, dim=0)

        # projection head
        outs = self.head(feature)  # (2n) * k

        # get items for loss
        if self.memory_bank.label_bank.is_cuda:
            cls_labels = self.memory_bank.label_bank[idx]
        else:
            cls_labels = self.memory_bank.label_bank[idx.cpu()].cuda()

        old_features = self.memory_bank.feature_bank[idx]
        new_features_centroids = self.memory_bank.centroids[cls_labels]

        # get random features and centroids
        memory_size = len(self.memory_bank.feature_bank)
        random_indices = random.sample(
            range(memory_size), self.sample_size)
        random_features = self.memory_bank.feature_bank[random_indices]
        random_features_labels = self.memory_bank.label_bank[random_indices]
        random_centroids = self.memory_bank.centroids[random_features_labels]

        # loss input
        loss_inputs = dict()
        loss_inputs['new_features_pairs'] = feature[0]
        loss_inputs['new_features_mean'] = mean_feature
        loss_inputs['old_features'] = old_features
        loss_inputs['new_prjections'] = outs
        loss_inputs['cls_labels'] = cls_labels
        loss_inputs['new_features_centroids'] = new_features_centroids
        loss_inputs['random_features'] = random_features
        loss_inputs['random_centroids'] = random_centroids

        # loss calculation
        losses = self.head.loss(*loss_inputs)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
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
