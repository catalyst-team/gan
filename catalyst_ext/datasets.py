"""
Implementations of metrics are based on
https://github.com/xuqiantong/GAN-Metrics/blob/master/metric.py
"""
import torchvision

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
