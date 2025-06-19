#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from utils.sampling import *

class DatasetLoader(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.data_distribution = args.iid
        self.num_user = args.num_user
        self.__load_dataset()
        self.__initial()

    # initial self.train_dataset and self.test_dataset
    def __load_dataset(self,path = "data/dataset"):
        # dataset path
        if self.dataset == 'MNIST':
            train_dataset = datasets.MNIST(path,
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

            test_dataset = datasets.MNIST(path,
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        elif self.dataset == 'CIFAR10':
            train_dataset = datasets.CIFAR10(path,
                                             train=True,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.RandomCrop(32, 4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010))
                                             ]))
            test_dataset = datasets.CIFAR10(path,
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010))
                                            ]))

        elif self.dataset == "FMNIST":
            train_dataset = datasets.FashionMNIST(path,
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                  ]))
            test_dataset = datasets.FashionMNIST(path,
                                                 train=False,
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))

        elif self.dataset == 'EMNIST':
            train_dataset = datasets.EMNIST(path,
                                            split="byclass",
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
            test_dataset = datasets.EMNIST(path,
                                           split="byclass",
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

        elif self.dataset == 'CIFAR100':
            train_dataset = datasets.CIFAR100(path,
                                              train = True,
                                              download = True,
                                              transform=transforms.Compose([
                                                    transforms.RandomCrop(32, 4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                      (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                                ]))
            test_dataset = datasets.CIFAR100(path,
                                             train = False,
                                             download = True,
                                             transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                                             ]))
        else:
            raise RuntimeError('the name inputed is wrong!')

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    # 初始化
    def __initial(self):
        if self.data_distribution == "IID":
            self.dict_train = iid(self.train_dataset, self.args.num_user)
            self.dict_test = iid(self.test_dataset, self.args.num_user)
        elif self.data_distribution == "Dir":
            self.dict_train = dirichlet_noniid(self.train_dataset, self.args.num_user, self.args.beta)
            self.dict_test = dirichlet_noniid(self.test_dataset, self.args.num_user, self.args.beta)
        elif self.data_distribution == "nonIID":         
            if self.dataset == 'MNIST':
                self.dict_train = mnist_noniid(self.train_dataset, self.args.num_user)
                self.dict_test = mnist_noniid(self.test_dataset, self.args.num_user)
            elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                self.dict_train = cifar_noniid(self.train_dataset, self.args.num_user)
                self.dict_test = cifar_noniid(self.test_dataset, self.args.num_user)
        else:
            raise RuntimeError('the name of data split inputed is wrong!')

    # get IID datasets
    def get_data(self, idxs):
        self.ldr_train = DataLoader(DatasetSplit(self.train_dataset, self.dict_train[idxs]), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(self.test_dataset, self.dict_test[idxs]), batch_size=self.args.local_bs, shuffle=True)
        return [self.ldr_train, self.ldr_test]

    def get_test_dataset(self):
        merged_array = list(self.dict_test.values())
        merged_array = [list(subset) for subset in merged_array]
        glob_dict_test = np.concatenate(merged_array)
        glob_test = DataLoader(DatasetSplit(self.test_dataset, glob_dict_test), batch_size=self.args.bs, shuffle=True)
        return glob_test

    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
