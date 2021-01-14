import torch
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class ContrastiveODCDatasetV2(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, for_extractor=False):
        super(ContrastiveODCDatasetV2, self).__init__(data_source, pipeline)
        # init clustering labels
        self.labels = [-1 for _ in range(self.data_source.get_length())]
        self.for_extractor = for_extractor

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        label = self.labels[idx]

        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        if self.for_extractor:
            img = self.pipeline(img)
            return dict(img=img, pseudo_label=label, idx=idx)
        else:
            # transform
            img1 = self.pipeline(img)
            img2 = self.pipeline(img)
            img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)

            return dict(img=img_cat, pseudo_label=label, idx=idx)

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent lenght of asigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

    def evaluate(self, scores, keyword, logger=None):

        raise NotImplemented
