#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import copy
import torch

from collections import defaultdict
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize

from utils.spectral_model import *

MODEL_PATH = './models/cache/'


class CloudServer(object):
    
    def __init__(self,model,device,args) -> None:
        self.__dummy_model = copy.deepcopy(model.to(device))
        self.__args = args
        self.__models = []
        self.__client_in_cluster = dict(int, list)
        self.__cluster_with_clients = dict(int, list)


    def secureCluster(self, models) -> dict:
        self.__models = models
        self.__client_in_cluster.clear()
        self.__cluster_with_clients.clear()
        clu_data = syn_model(self.__models)
        sz = len(self.__models)
        pairwise_matrix = rbf_kernel(clu_data, clu_data, 0.01)
        spectral = SpectralClustering(n_clusters=self.__args.c, affinity='precomputed', assign_labels="discretize", random_state=0)
        self.labels = spectral.fit_predict(pairwise_matrix)
        self.c = len(set(self.labels))
        _, eigenvectors = eigh(pairwise_matrix)
        U = eigenvectors[:, -self.c:]
        U_normalized = normalize(U)
        memberships = np.zeros((sz, self.c))
        for i in range(self.c):
            cluster_center = U_normalized[self.labels == i]
            if cluster_center.any():
                cluster_center_mean = cluster_center.mean(axis=0)
                memberships[:, i] =np.sqrt(np.sum((U_normalized - cluster_center_mean)** 2, axis=1)) 
        memberships = normalize(memberships, norm='l1', axis=1)
        thresholds = np.nanmean(memberships, axis=1, keepdims=True)
        sample_cluster_memberships = np.where(np.isnan(memberships), 0, np.where(memberships >= thresholds, 1, 0))
        assert(np.all(sample_cluster_memberships.sum(axis=1)>0))
        for shuffled_id, row in enumerate(sample_cluster_memberships):  
            if np.sum(row)==0 : continue
            # For distribution
            self.__client_in_cluster[shuffled_id] = list(np.where(row==1)[0])
        for cluster_id, col in enumerate(sample_cluster_memberships.T):
            # For clustering
            self.__cluster_with_clients[cluster_id] = list(col)
        return self.__client_in_cluster, self.__cluster_with_clients


    # def aggregateLocalModelsInCluster(self,participants):

    #     self.cluster_models = [copy.deepcopy(self._model.state_dict())]

    #     for j, cluster_model in enumerate(self.cluster_models):
    #         for key in cluster_model.keys():
    #             cluster_model[key] *= 0.0
    #             count = 0
    #             for participant in participants:
    #                 if participant._label == j: 
    #                     m = participant._model.state_dict()
    #                     cluster_model[key] += m[key]
    #                     count += 1
    #             cluster_model[key] = torch.div(cluster_model[key], count)
    #         torch.save(cluster_model, MODEL_PATH+'cluster_model_{}.pt'.format(j))


    def aggregateLocalModelsInSoftCluster(self) -> dict:

        self.dic_agg_results = dict()
        for clu_id, shuffled_id in enumerate(self.__cluster_with_clients.items()):
            size = len(shuffled_id)
            aggregated_candidates = [self.__models[i].state_dict() for i in shuffled_id]
            cluster_aggregated_res = copy.deepcopy(aggregated_candidates[0].state_dict())
            for m in aggregated_candidates:
                for key in cluster_aggregated_res.keys():
                    cluster_aggregated_res[key] += m[key]
                cluster_aggregated_res[key] =  torch.div(cluster_aggregated_res[key], size)
            self.dic_agg_results[clu_id] = cluster_aggregated_res
        # print(self.agg_results.keys())
        return self.dic_agg_results


    def aggregateClusterModels(self):
        
        global_weight = copy.deepcopy(self.dic_agg_results[0])
        
        for key in global_weight.keys():
            for i in range(1, self.c):
                global_weight[key] += self.dic_agg_results[i][key]
            global_weight[key] = torch.div(global_weight[key], self.c)

        self.__dummy_model.load_state_dict(global_weight)


    # def evalGlobalModel(self):

    #     for c_model in self.agg_results.values():
    #         test_loss = math.inf
    #         test_correct = 0.0
    #         cur_loss = 0.0
    #         cur_correct = 0.0
    #         self._model.load_state_dict(c_model)
    #         self._model.eval()
    #         with torch.no_grad():
    #             for data, target in self._testset:
    #                 data, target = Variable(data).to(self._device), Variable(target).to(self._device)
    #                 output = self._model(data) 
    #                 cur_loss += torch.nn.CrossEntropyLoss()(output, target).item()
    #                 cur_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()
    #             test_correct = max(test_correct, cur_correct/len(self._testset.dataset))
    #             test_loss = min(test_loss, cur_loss/len(self._testset.dataset))

    #     return test_correct, test_loss
    

