from __future__ import annotations
from enum import Enum
import os
import json

import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
from numpy.random import permutation
from fl_bench.data import DataContainer, FastTensorDataLoader
from fl_bench.data.dataclass import SVHN


class Datasets:
    """Static class for loading datasets.

    Each dataset is loaded as a DataContainer object.
    """

    @classmethod
    def MNIST(cls) -> DataContainer:
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

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)
    
    @classmethod
    def MNIST4D(cls) -> DataContainer:
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

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = train_data.data[:, None, :, :]  # added because probably without the transformation it does not have the correct number of dimension
                                                          # in this way the dimension to the convolutional layer are correct (64,1,28,28)
        test_data.data = test_data.data[:, None, :, :]

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)
    
    @classmethod
    def MNISTM(cls) -> DataContainer:
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
        return DataContainer(train_data.data / 255., 
                             train_data.targets, 
                             test_data.data / 255., 
                             test_data.targets, 
                             10)
    
    @classmethod
    def EMNIST(cls) -> DataContainer:
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
        return DataContainer(train_data.data / 255.,
                             train_data.targets, 
                             test_data.data / 255., 
                             test_data.targets, 
                             26)
    
    @classmethod
    def SVHN(cls) -> DataContainer:
        train_data = SVHN(
            root = 'data',
            train = True,
            download = True
        )

        test_data = SVHN(
            root = 'data',
            train = False,
            download = True
        )
        return DataContainer(train_data.data / 255., 
                             train_data.targets, 
                             test_data.data / 255.,
                             test_data.targets, 
                             10)

    @classmethod
    def CIFAR10(cls) -> DataContainer:
        train_data = datasets.CIFAR10(
            root = 'data',
            train = True,
            download = True, 
            transform = None
        )

        test_data = datasets.CIFAR10(
            root = 'data',
            train = False,
            download = True, 
            transform = None
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
        return Datasets.SVMLIGHT(filename_tr, filename_te, 0.2, 42)
    
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
    def FEMNIST(cls, batch_size=10):

        assert os.path.exists('data/FEMNIST'), "FEMNIST data ('data/FEMNIST') not found."
        assert os.path.exists('data/FEMNIST/train'), "FEMNIST train data ('data/FEMNIST/train') not found."
        assert os.path.exists('data/FEMNIST/test'), "FEMNIST test data ('data/FEMNIST/test') not found."

        # TRAINING
        train_dir = 'data/FEMNIST/train'
        files = os.listdir(train_dir)
        dict_train = {}
        for file in files:
            with open(os.path.join(train_dir, file)) as f:
                data = json.load(f)  
            dict_train.update(data["user_data"])

        # TEST
        test_dir = 'data/FEMNIST/test'
        files = os.listdir(test_dir)
        dict_test = {}
        for file in files:
            with open(os.path.join(test_dir, file)) as f:
                data = json.load(f)
            dict_test.update(data["user_data"])

        client_tr_assignments = []
        for udata in dict_train.values():
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
        for udata in dict_test.values():
            Xte_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            yte_client = torch.LongTensor(udata["y"])
            client_te_assignments.append(
                FastTensorDataLoader(
                    Xte_client, 
                    yte_client, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    percentage=1.0
                )
            )
        
        return client_tr_assignments, client_te_assignments, None

class DatasetsEnum(Enum):
    MNIST = "mnist"
    MNIST4D = "mnist4d"
    MNISTM = "mnistm"
    SVHN = "svhn"
    FEMNIST = "femnist"
    EMNIST = "emnist"
    CIFAR10 = "cifar10"
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

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, DatasetsEnum):
            return member.value in cls._member_names_

    def klass(self):
        DATASET_MAP = {
            "mnist": Datasets.MNIST,
            "mnistm": Datasets.MNISTM,
            "mnist4d": Datasets.MNIST4D,
            "svhn": Datasets.SVHN,
            "femnist": Datasets.FEMNIST,
            "emnist": Datasets.EMNIST,
            "cifar10": Datasets.CIFAR10,
            "letter": Datasets.LETTER,
            "pendigits": Datasets.PENDIGITS,
            "satimage": Datasets.SATIMAGE,
            "forestcover": Datasets.FORESTCOVER,
            "segmentation": Datasets.SEGMENTATION,
            "adult": Datasets.ADULT,
            "krvskp": Datasets.KRVSKP,
            "splice": Datasets.SPLICE,
            "vehicle": Datasets.VEHICLE,
            "vowel": Datasets.VOWEL
        } 
        return DATASET_MAP[self.value]

    def splitter(self):
        from . import DataSplitter, DummyDataSplitter

        if self.value == "femnist":
            return DummyDataSplitter
        else:
            return DataSplitter

