import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class ContrastiveODCDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveODCDataset, self).__init__(data_source, pipeline)
        # init clustering labels
        self.labels = [-1 for _ in range(self.data_source.get_length())]

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        # transform
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat, pseudo_label=label, idx=idx)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
