#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
import math
import torch
import numpy as np

from torch.autograd import Variable



MODEL_PATH = './models/cache/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

class ServiceProvider(object):
    
    def __init__(self,args, clients) -> None:
        self.args = args
        self.__models = []
        self.__blind_factors = {}
        self.__participants = clients
        
    def blindModel(self) -> None:
        for p in self.__participants:
            client_model = copy.deepcopy(torch.load(MODEL_PATH+'client_model_{}.pt'.format(p.id)))
            for identifier, param in client_model.named_parameters():
                # Generate a random blind factor for each parameter
                blind_factor = torch.randn_like(param.data)
                self.__blind_factors[identifier] = blind_factor
                # Blind the model parameters
                param.data += blind_factor
            self.__models.append(client_model)
    
    def shuffleIdentifier(self) -> list:
        n = len(self.__models)
        self.__permutated_identifiers = np.random.permutation(n)
        self.__blind_models = [self.__models[i] for i in self.__permutated_identifiers]
        return self.__blind_models
    
    def computeFederatedGradientNorm(self) -> float:
        fgn = 0.0
        for p in self.__participants:
            fgn += p.m * p.lgn
        return fgn
    
    def unblindClusterModel(self, clu_models, clu, cli) -> object:
        self.cluster_models = clu_models
        for clients in clu.values():
            for client_id in clients:
                if client_id in cli:
                    for identifier, param in self.cluster_models[client_id].named_parameters():
                        # Unblind the model parameters
                        param.data -= self.__blind_factors[identifier]
        return self.cluster_models


    def evalGlobalModel(self):
        for c_model in self.cluster_models.values():
            test_loss = math.inf
            test_correct = 0.0
            cur_loss = 0.0
            cur_correct = 0.0
            self._model.load_state_dict(c_model)
            self._model.eval()
            with torch.no_grad():
                for data, target in self._testset:
                    data, target = Variable(data).to(self._device), Variable(target).to(self._device)
                    output = self._model(data) 
                    cur_loss += torch.nn.CrossEntropyLoss()(output, target).item()
                    cur_correct += (torch.sum(torch.argmax(output,dim=1)==target)).item()
                test_correct = max(test_correct, cur_correct/len(self._testset.dataset))
                test_loss = min(test_loss, cur_loss/len(self._testset.dataset))

        return test_correct, test_loss
    


   