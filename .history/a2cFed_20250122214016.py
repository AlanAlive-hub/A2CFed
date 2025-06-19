#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7
import random
import time
import math
import copy
import numpy as np
import torch
import pandas as pd

from client import Client
from cloud_service import CloudServer

from models.model import getModel
from utils.data_loader import DatasetLoader
from utils.options import args_parser


class FedPCA():
    def __init__(self, args) -> None:
        self.args = args

    def setup(self):
        
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        self.device = torch.device('cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')

    def init_entity(self):
        
        model = getModel(args=self.args, device=self.device)
        data_loader = DatasetLoader(args=self.args)

        self._cloud = CloudServer(model=copy.deepcopy(model),device=self.device,dataset=data_loader.get_test_dataset(),args=self.args)
        self._clients = [Client(id=idx,model=copy.deepcopy(model),device=self.device,dataset=data_loader.get_data(idx),args=self.args) 
                        for idx in range(self.args.num_user)]
        # self._kgc = KeyGenerateCenter(len(self._clients))
        # self._service_provider = ServiceProvider(model=copy.deepcopy(model), device=self.device, args=self.args)
        for client in self._clients : client.pre_train()

    def training(self, alpha=None):
        results = []
        # num_participant_client = max(int(self.args.frac * len(self._clients)), 1)
        marks = [0] * self.args.epoch
        for ep in range(self.args.epoch):
            # participants = []
            local_accuracies= []
            start_time = time.time()
            
            for client in self._clients:
                client._delay = float(random.randint(1,self.args.num_user)/self.args.num_user)
            # self._clients.sort(key=lambda cli: cli._delay)
            # participants = self._clients[:num_participant_client]

            # offset =  int(math.ceil(self.args.num_user / self.args.k1))
            # clients_groups = [self._clients[i:i+offset] for i in range(0, self.args.num_user, offset)]
            # for cli_g in clients_groups:
            #     num_to_remove = math.ceil(len(cli_g) * 0.1)
            #     for cli in cli_g:
            #         cli.trainLocalModel()
            #     trimmed_group = cli_g[num_to_remove:]
            #     for c in trimmed_group:
            #         participants.append(c)
            for cli in self._clients:
                cli.trainLocalModel()
            if random.random() <= 1/(1+ep*alpha) or ep == 0:
                marks[ep] = 1
            # if ep % alpha == 0:
                self._cloud.secureClusterLocalModels(self._clients)
            glob_weights = self._cloud.aggregateLocalModelsInSoftCluster()
            # self._cloud.aggregateClusterModels()
            glob_acc, glob_loss = self._cloud.evalGlobalModel()

            for cli in self._clients:
                cli.updateLocalModel(glob_weights)
                acc = cli.evalLocalModel()
                local_accuracies.append(acc)

            epoch_time  = time.time() - start_time

            mean_local_accuracies = np.mean(local_accuracies)
            print("Round {:3d}, Local Testing accuracy: {:.4f}, Glob Testing accuracy: {:.4f}, training time:{:.3f}"
                                .format(ep+1, mean_local_accuracies, glob_acc, epoch_time))
            
            results.append({
                "dataset": self.args.dataset,
                "average_client_accuracy": mean_local_accuracies,
                "glob_accuracy":glob_acc,
                "glob_loss":glob_loss,
                "training time": epoch_time,
                "clusert mark": marks[ep]
            })
            
        return results, marks.count(1)
    

conf = {
    #  'MNIST':(0.8, 1, 50),
    # 'FMNIST':(0.8, 1, 200),
    # 'FMNIST':(0.8, 1, 200),
    'CIFAR10':(0.3, 1, 200),
    # 'CIFAR100':(0.8, 1, 200),
}

for dataset, (beta, pre_epoch, epoch) in conf.items():
    args = args_parser()
    args.dataset = dataset
    args.pre_epoch = pre_epoch
    args.c = 30
    args.beta = beta
    args.epoch = epoch
    print(f"fedpac_results_{args.dataset}_Dir{args.beta}_{args.c}")
    fedpac = FedPCA(args=args)
    fedpac.setup()
    fedpac.init_entity()
    results, counter = fedpac.training(alpha=0.1)
    # 保存结果到Excel
    df = pd.DataFrame(results)
    df.to_excel(f"fedpac_results_{args.dataset}_Dir{args.beta}_{args.c}_{counter}_{0.1}.xlsx", index=False)