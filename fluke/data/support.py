"""This module contains utility classes for loading datasets."""
import os
import sys
from typing import Any

import torch
from PIL import Image
from rich.progress import track
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor

sys.path.append(".")
sys.path.append("..")

__all__ = [
    "MNISTM",
    "CINIC10"
]


class MNISTM(VisionDataset):
    """MNIST-M Dataset.
    """

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        data_file = (self.training_file if self.train
                     else self.test_file)
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)


class CINIC10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset. CINIC-10 is an augmented
    extension of CIFAR-10. It contains the images from CIFAR-10 (60,000 images, 32x32 RGB pixels)
    and a selection of ImageNet database images (210,000 images downsampled to 32x32). It was
    compiled as a 'bridge' between CIFAR-10 and ImageNet, for benchmarking machine learning
    applications. It is split into three equal subsets - train, validation, and test -
    each of which contain 90,000 images.

    Args:
        root (string): Root directory of dataset where directory ``cifar-10-batches-py`` exists
            or will be saved to if download is set to True.
        partition (str, optional): One of train,valid,test, creates selects which partition to
            use.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it
            in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    base_folder = "CINIC10"
    filename = "CINIC-10.tar.gz"
    url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    file_md5 = "6ee4d0c996905fe93221de577967a372"
    # mean = [0.47889522, 0.47227842, 0.43047404]
    # std = [0.24205776, 0.23828046, 0.25874835]

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    def __init__(self,
                 root: str = "../data",
                 split: str = "train",
                 #  transform: Optional[Callable] = None,
                 download: bool = True):

        super().__init__(root)
        self.split = split
        self.root = os.path.join(root, "CINIC10")

        if download:
            self.download()

        self._dataset = ImageFolder(os.path.join(self.root, split))
        self.data = self._images2tensor()
        self.targets = torch.LongTensor(self._dataset.targets)

    def _images2tensor(self):
        img_tensors = []
        for img, _ in track(self._dataset.imgs, f"Loading CINIC-10 {self.split} set..."):
            img_tensor = ToTensor()(Image.open(img))
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            img_tensors.append(img_tensor)

        return torch.stack(img_tensors)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        if os.path.isdir(os.path.join(self.root, self.split)):
            return
        else:
            download_and_extract_archive(self.url,
                                         download_root=self.root,
                                         md5=CINIC10.file_md5)

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(CINIC10.classes)}
