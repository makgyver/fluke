from __future__ import annotations
import os
import json
import torch
import numpy as np
import pandas as pd
from enum import Enum

from numpy.random import permutation

from rich.progress import track

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from fl_bench.data import DataContainer, FastTensorDataLoader, support


class Datasets:
    """Static class for loading datasets.

    Each dataset is loaded as a DataContainer object.
    """

    @classmethod
    def MNIST(cls, 
              path: str="../data", 
              transforms: callable=ToTensor) -> DataContainer:
        
        train_data = datasets.MNIST(
            root = path,
            train = True,                         
            transform = transforms(), 
            download = True,            
        )

        test_data = datasets.MNIST(
            root = path, 
            train = False, 
            transform = transforms(),
            download = True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)
    
    @classmethod
    def MNIST4D(cls,
                path: str="../data", 
                transforms: callable=ToTensor) -> DataContainer:
        
        mnist_dc = Datasets.MNIST(path, transforms)
        return DataContainer(mnist_dc.train[0][:, None, :, :], 
                             mnist_dc.train[1],
                             mnist_dc.test[0][:, None, :, :], 
                             mnist_dc.test[1], 
                             10)
    
    @classmethod
    def MNISTM(cls,
               path: str="../data", 
               transforms: callable=ToTensor) -> DataContainer:
        
        train_data = support.MNISTM(
            root = path,
            train = True,                         
            transform = transforms(), 
            download = True,            
        )

        test_data = support.MNISTM(
            root = path, 
            train = False, 
            transform = transforms(),
            download = True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)

    @classmethod
    def EMNIST(cls,
               path: str="../data", 
               transforms: callable=ToTensor) -> DataContainer:
        
        train_data = datasets.EMNIST(
            root=path,
            split="balanced",
            train=True, 
            transform=transforms(),
            download = True
        )

        test_data = datasets.EMNIST(
            root=path,
            split="balanced", 
            train=False,
            transform=transforms(),
            download = True
        )

        return DataContainer(train_data.data / 255.,
                             train_data.targets, 
                             test_data.data / 255., 
                             test_data.targets, 
                             47)
    
    @classmethod
    def SVHN(cls,
             path: str="../data", 
             transforms: callable=ToTensor) -> DataContainer:
        
        train_data = datasets.SVHN(
            root = path,
            split = "train",
            transform=transforms(),
            download = True
        )

        test_data = datasets.SVHN(
            root = path,
            split = "test",
            transform=transforms(),
            download = True
        )

        return DataContainer(train_data.data / 255., 
                             train_data.labels, 
                             test_data.data / 255.,
                             test_data.labels, 
                             10)

    @classmethod
    def CIFAR10(cls,
                path: str="../data", 
                transforms: callable=ToTensor) -> DataContainer:
        
        train_data = datasets.CIFAR10(
            root = path,
            train = True,
            download = True, 
            transform = transforms()
        )

        test_data = datasets.CIFAR10(
            root = path,
            train = False,
            download = True, 
            transform = transforms()
        )
        
        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data, 
                             torch.LongTensor(train_data.targets), 
                             test_data.data, 
                             torch.LongTensor(test_data.targets), 
                             10)

    @classmethod
    def CIFAR100(cls,
                 path: str="../data", 
                 transforms: callable=ToTensor) -> DataContainer:
    
        train_data = datasets.CIFAR100(
            root = path,
            train = True,
            download = True, 
            transform = transforms()
        )

        test_data = datasets.CIFAR100(
            root = path,
            train = False,
            download = True, 
            transform = transforms()
        )
        
        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data, 
                             torch.LongTensor(train_data.targets), 
                             test_data.data, 
                             torch.LongTensor(test_data.targets), 
                             100)


    @classmethod
    def TINY_IMAGENET(cls, 
                      path: str="../data", 
                      transforms: callable=None) -> DataContainer:
        
        tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', 
                                           split='train', 
                                           cache_dir=path)

        tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', 
                                          split='valid', 
                                          cache_dir=path)

        X_train, y_train = [], []
        X_test, y_test = [], []

        fix_bw_trans = Lambda(lambda x: x.repeat(1, 3, 1, 1))

        for image in track(tiny_imagenet_train, "Loading Tiny ImageNet train data..."):
            y = image['label']
            image = image['image']
            x = ToTensor()(image).unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_train.append(x)
            y_train.append(y)

        for image in track(tiny_imagenet_test, "Loading Tiny ImageNet test data..."):
            y = image['label']
            image = image['image']
            x = ToTensor()(image).unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_test.append(x)
            y_test.append(y)

        train_data = torch.vstack(X_train)
        test_data = torch.vstack(X_test)

        if transforms is not None:
            train_data = transforms(train_data)
            test_data = transforms(test_data)

        idxperm = torch.randperm(train_data.shape[0])
        train_data = train_data[idxperm]
        y_train = torch.LongTensor(y_train)[idxperm]
        y_test = torch.LongTensor(y_test)

        return DataContainer(train_data, 
                             y_train, 
                             test_data, 
                             y_test, 
                             200)

    @classmethod
    def LETTER(cls, filename: str="data/letter.csv", test_size: float=0.2, seed: int=42) -> DataContainer:
        df = pd.read_csv(filename, header=None)
        feats = ["feat_%d" % i for i in range(df.shape[1]-1)]
        df.columns = ["label"] + feats
        X = df[feats].to_numpy()
        y = LabelEncoder().fit_transform(df["label"].to_numpy())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        return DataContainer(X_train, y_train, X_test, y_test, 26)

    @classmethod
    def PENDIGITS(cls, filename_tr: str="data/pendigits.tr.csv", filename_te: str="data/pendigits.te.csv") -> DataContainer:
        df_tr = pd.read_csv(filename_tr, header=None)
        df_te = pd.read_csv(filename_te, header=None)
        y_tr = df_tr.loc[:, 16].to_numpy()
        y_te = df_te.loc[:, 16].to_numpy()
        X_tr = df_tr.loc[:, :15].to_numpy()
        X_te = df_te.loc[:, :15].to_numpy()
        return DataContainer(X_tr, y_tr, X_te, y_te, 10)

    @classmethod
    def SATIMAGE(cls, filename_tr: str="data/sat.tr.csv", filename_te: str="data/sat.te.csv") -> DataContainer:
        df_tr = pd.read_csv(filename_tr, sep=" ", header=None)
        df_te = pd.read_csv(filename_te, sep=" ", header=None)
        y_tr = df_tr.loc[:, 36].to_numpy()
        y_te = df_te.loc[:, 36].to_numpy()
        X_tr = df_tr.loc[:, :35].to_numpy()
        X_te = df_te.loc[:, :35].to_numpy()
        le = LabelEncoder().fit(y_tr)
        y_tr = le.transform(y_tr)
        y_te = le.transform(y_te)
        return DataContainer(X_tr, y_tr, X_te, y_te, 6)

    @classmethod
    def FORESTCOVER(cls, filename: str="data/covtype.data"):
        covtype_df = pd.read_csv(filename, header=None)
        covtype_df = covtype_df[covtype_df[54] < 3]
        X = covtype_df.loc[:, :53].to_numpy()
        y = (covtype_df.loc[:, 54] - 1).to_numpy()
        ids = permutation(X.shape[0])
        X, y = X[ids], y[ids]
        X_train, X_test = X[:250000], X[250000:]
        y_train, y_test = y[:250000], y[250000:]
        return DataContainer(X_train, y_train, X_test, y_test, 2)

    @classmethod
    def SVMLIGHT(cls, filename_tr: str, filename_te: str=None, test_size: float=0.2, seed: int=42):
        if not filename_te is None:
            X_tr, y_tr = load_svmlight_file(filename_tr)
            X_te, y_te = load_svmlight_file(filename_te)
            X_tr = X_tr.toarray()
            X_te = X_te.toarray()
            
        else:
            X, y = load_svmlight_file(filename_tr)
            X = X.toarray()
            y = y.astype("int")
            y = LabelEncoder().fit_transform(y)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        num_classes = len(np.unique(y_tr))
        return DataContainer(X_tr, y_tr, X_te, y_te, num_classes=num_classes)
    
    @classmethod
    def SEGMENTATION(cls, filename_tr="data/segmentation.tr.svmlight", filename_te="data/segmentation.te.svmlight", test_size=0.2, seed=42):
        return Datasets.SVMLIGHT(filename_tr, filename_te, test_size, 42)
    
    @classmethod
    def ADULT(cls, filename_tr="data/adult.tr.svmlight", filename_te="data/adult.te.svmlight", test_size=0.2, seed=42):
        return Datasets.SVMLIGHT(filename_tr, filename_te, test_size, seed)

    @classmethod
    def KRVSKP(cls, filename: str="data/kr-vs-kp.svmlight", test_size: float=0.2, seed: int=42):
        return Datasets.SVMLIGHT(filename, None, test_size, seed)

    @classmethod
    def SPLICE(cls, filename: str="data/splice.svmlight", test_size: float=0.2, seed: int=42):
        return Datasets.SVMLIGHT(filename, None, test_size, seed)

    @classmethod
    def VEHICLE(cls, filename: str="data/vehicle.svmlight", test_size: float=0.2, seed: int=42):
        return Datasets.SVMLIGHT(filename, None, test_size, seed)

    @classmethod
    def VOWEL(cls, filename: str="data/vowel.svmlight", test_size: float=0.2, seed: int=42):
        return Datasets.SVMLIGHT(filename, None, test_size, seed)


    @classmethod
    def FEMNIST(cls, 
                path: str="./data",
                batch_size: int=10):

        femnist_path = os.path.join(path, "FEMNIST")
        train_dir = os.path.join(femnist_path, 'train')
        test_dir = os.path.join(femnist_path, 'test')

        assert os.path.exists(femnist_path), f"FEMNIST data ({femnist_path}) not found."
        assert os.path.exists(train_dir), f"FEMNIST train data ({train_dir}') not found."
        assert os.path.exists(test_dir), f"FEMNIST test data ({test_dir}') not found."

        # TRAINING
        files = os.listdir(train_dir)
        dict_train = {}
        for file in track(files, "Loading FEMNIST train data..."):
            with open(os.path.join(train_dir, file)) as f:
                data = json.load(f)  
            dict_train.update(data["user_data"])

        # TEST
        files = os.listdir(test_dir)
        dict_test = {}
        for file in track(files, "Loading FEMNIST test data..."):
            with open(os.path.join(test_dir, file)) as f:
                data = json.load(f)
            dict_test.update(data["user_data"])
            
        client_tr_assignments = []
        for k in track(sorted(dict_train), "Creating training data loader..."):
            udata = dict_train[k]
            Xtr_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            ytr_client = torch.LongTensor(udata["y"])
            client_tr_assignments.append(
                FastTensorDataLoader(
                    Xtr_client, 
                    ytr_client, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    percentage=1.0
                )
            )
        
        client_te_assignments = []
        for k in track(sorted(dict_train), "Creating testing data loader..."):
            udata = dict_test[k]
            Xte_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            yte_client = torch.LongTensor(udata["y"])
            client_te_assignments.append(
                FastTensorDataLoader(
                    Xte_client, 
                    yte_client, 
                    batch_size=64, 
                    shuffle=True, 
                    percentage=1.0
                )
            )
            
        perm = permutation(len(client_tr_assignments))
        client_tr_assignments = [client_tr_assignments[i] for i in perm]
        client_te_assignments = [client_te_assignments[i] for i in perm]
        return client_tr_assignments, client_te_assignments, None


class DatasetsEnum(Enum):
    MNIST = "mnist"
    MNISTM = "mnistm"
    MNIST4D = "mnist4d"
    SVHN = "svhn"
    FEMNIST = "femnist"
    EMNIST = "emnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    LETTER = "letter"
    PENDIGITS = "pendigits"
    SATIMAGE = "satimage"
    FORESTCOVER = "forestcover"
    SEGMENTATION = "segmentation"
    ADULT = "adult"
    KRVSKP = "krvskp"
    SPLICE = "splice"
    VEHICLE = "vehicle"
    VOWEL = "vowel"
    TINY_IMAGENET = "tiny_imagenet"

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, DatasetsEnum):
            return member.value in cls._member_names_

    def klass(self):
        DATASET_MAP = {
            "mnist": Datasets.MNIST,
            "mnist4d": Datasets.MNIST4D,
            "svhn": Datasets.SVHN,
            "mnistm": Datasets.MNISTM,
            "femnist": Datasets.FEMNIST,
            "emnist": Datasets.EMNIST,
            "cifar10": Datasets.CIFAR10,
            "cifar100": Datasets.CIFAR100,
            "letter": Datasets.LETTER,
            "pendigits": Datasets.PENDIGITS,
            "satimage": Datasets.SATIMAGE,
            "forestcover": Datasets.FORESTCOVER,
            "segmentation": Datasets.SEGMENTATION,
            "adult": Datasets.ADULT,
            "krvskp": Datasets.KRVSKP,
            "splice": Datasets.SPLICE,
            "vehicle": Datasets.VEHICLE,
            "vowel": Datasets.VOWEL,
            "tiny_imagenet": Datasets.TINY_IMAGENET
        } 
        return DATASET_MAP[self.value]

    def splitter(self):
        from . import DataSplitter, DummyDataSplitter

        if self.value == "femnist":
            return DummyDataSplitter
        else:
            return DataSplitter

