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

from entity.client import Client
from entity.cloud_service import CloudServer
from entity.business_server import ServiceProvider
from entity.crypto_generate_center import KeyGenerateCenter

from utils.model import getModel
from utils.data_loader import DatasetLoader
from utils.options import args_parser


class CAPFed():
    def __init__(self, args) -> None:
        self.args = args
        self.setup()
        self.init_entity()
        

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

        self.clients = [Client(id=idx,model=copy.deepcopy(model),device=self.device,dataset=data_loader.get_data(idx),args=self.args) 
                        for idx in range(self.args.num_user)]
        self.cloud_service_provider = CloudServer(model=copy.deepcopy(model),device=self.device,args=self.args)
        self.business_server = ServiceProvider(args=self.args, clients=self.clients)
        # self._kgc = KeyGenerateCenter(len(self.clients))

    def training(self, alpha=0.75, lmbda=0.65):
        local_accuracies, results, clustering_mark = [], [], 0
        for client in self.clients : client.preTrain()
        for ep in range(self.args.epoch):
            start_time = time.time()
            # local training
            for cli in self.clients:
                cli.localTraining()
                cli.computeLocalGradientNorm()
            # BS side
            self.business_server.blindModel()
            anonymous_data = self.business_server.shuffleIdentifier()
            fgn = self.business_server.computeFederatedGradientNorm()
            # CSP side
            t = (1-alpha)/(alpha*lmbda)
            gamma = ep / (self.args.epoch - 1)
            lower = round(fgn * 10) / 10 - 0.1
            upper = lower + 0.1
            pr = alpha * math.exp(-gamma/alpha) * (1 - math.exp(-(1/ (self.args.epoch - 1))/alpha)) * (t*(upper-lower) + (1-t) * ((upper**lmbda-lower**lmbda)/(lmbda+1)))
            if random.uniform(0,0.02) <= pr or ep == 0:
                cli_in_clu, clu_with_cli = self.cloud_service_provider.secureCluster(anonymous_data)
                clustering_mark = 1
            dic_cluster_models = self.cloud_service_provider.aggregateLocalModelsInSoftCluster()
            # BS side
            update_models = self.business_server.unblindClusterModel(dic_cluster_models, cli_in_clu, clu_with_cli)
            glob_acc, glob_loss = self.business_server.evalGlobalModel()
            # update local models
            for cli in self.clients:
                cli.updateLocalModel(update_models)
                l_acc = cli.evalLocalModel()
                local_accuracies.append(l_acc)
            epoch_time  = time.time() - start_time
            mean_local_accuracies = np.mean(local_accuracies)
            print("Round {:3d}, Local Testing accuracy: {:.4f}, Glob Testing accuracy: {:.4f}, training time:{:.3f}, Clustering Mark: {}".format(
                ep+1, mean_local_accuracies, glob_acc, epoch_time, clustering_mark))
            local_accuracies.clear()
            clustering_mark = 0
            # save results
            results.append({
                "dataset": self.args.dataset,
                "average_client_accuracy": mean_local_accuracies,
                "glob_accuracy":glob_acc,
                "glob_loss":glob_loss,
                "training time": epoch_time,
                "cluserting mark": clustering_mark
            })
        return results

def __main__():
    args = args_parser()
    capfed = CAPFed(args=args)
    results, counter = capfed.training()
    df = pd.DataFrame(results)
    df.to_excel(f"exp_{args.dataset}_Dir{args.beta}_{args.c}_{counter}_{0.1}.xlsx", index=False)