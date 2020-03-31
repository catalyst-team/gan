"""
Custom datasets
"""
import os

import torch.utils.data
import torchvision
import cv2

from catalyst.dl import registry

DATASETS = registry.Registry("dataset")
Dataset = DATASETS.add


@Dataset
class MNIST(torchvision.datasets.MNIST):
    """
    MNIST Dataset with key_value __get_item__ output
    """
    def __init__(
        self,
        root='./data/mnist',
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        image_key="image",
        target_key="target"
    ):
        """

        :param root:
        :param train:
        :param transform:
        :param target_transform:
        :param download:
        :param image_key: key to place an image
        :param target_key: key to place target
        """
        super().__init__(root, train, transform, target_transform, download)
        self.image_key = image_key
        self.target_key = target_key

    def __getitem__(self, index: int):
        """Get dataset element"""
        image, target = self.data[index], self.targets[index]

        dict_ = {
            self.image_key: image,
            self.target_key: target,
        }

        if self.transform is not None:
            dict_ = self.transform(dict_)
        return dict_


@Dataset
class ImageOnlyDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None,
                 image_key: str = "image",
                 filename_key: str = "filename",
                 cache: bool = False):
        self.root_dir = root_dir
        self.transform = transform

        self.image_key = image_key
        self.filename_key = filename_key

        self.image_names = os.listdir(self.root_dir)

        self.is_caching = cache
        self._cache = [None for _ in self.image_names]

    def __getitem__(self, index: int):
        if self.is_caching and self._cache[index] is not None:
            return self._cache[index]
        image_name = self.image_names[index]
        image = cv2.imread(os.path.join(self.root_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dict_ = {
            self.image_key: image,
            self.filename_key: image_name
        }
        if self.transform is not None:
            dict_ = self.transform(dict_)

        if self.is_caching:
            self._cache[index] = dict_
        return dict_

    def __len__(self) -> int:
        return len(self.image_names)
