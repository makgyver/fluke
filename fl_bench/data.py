from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets import MNIST, utils
from PIL import Image
import os
import os.path
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

class Datasets:

    @classmethod
    def MNIST(cls):
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor(),
            download = True
        )
        return train_data, test_data
    
    @classmethod
    def EMNIST(cls):
        train_data = datasets.EMNIST(
            root="data",
            split="balanced",
            train=True, 
            transform=ToTensor(),
            download = True
        )

        test_data = datasets.EMNIST(
            root="data",
            split="balanced", 
            train=False,
            transform=ToTensor(),
            download = True
        )
        return train_data, test_data
    
    @classmethod
    def FEMNIST(cls):
        train_data = FEMNIST(
            root="data",
            train=True, 
            transform=ToTensor(),
            download = True
        )

        test_data = FEMNIST(
            root="data",
            train=False,
            transform=ToTensor(),
            download = True
        )
        return train_data, test_data


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))
        self.data = self.data * 255.

    def _check_exists(self) -> bool:
        return all(
            utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url))) for url, _ in self.resources
        )
        
    # def __getitem__(self, index):
    #     img, target = self.data[index], int(self.targets[index])
    #     img = Image.fromarray(img.numpy(), mode='F')
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        training_file = os.path.join(self.raw_folder, self.training_file)
        if not os.path.isfile(training_file):
            shutil.move(training_file, self.processed_folder)
        test_file = os.path.join(self.raw_folder, self.test_file)
        if not os.path.isfile(training_file):
            shutil.move(test_file, self.processed_folder)



class MNISTDataset(Dataset):
    def __init__(self, data, left, right, transform=None, target_transform=None):
        self.left = left
        self.right = right if right > 0 else len(data.targets)
        self.data = data.data[self.left:self.right]
        self.targets = data.targets[self.left:self.right]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.right - self.left

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img/255., target