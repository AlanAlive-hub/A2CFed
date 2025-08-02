#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
import torch
from torch import nn, optim, Variable
from torch.autograd import Variable


MODEL_PATH = './models/cache/'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


class Client(object):
    def __init__(self, id, model, device, dataset, args):
        self.args = args
        self.__id = id
        self.__device = device
        self.__delay = 0.0
        self.__lossfunc = nn.CrossEntropyLoss().to(self.__device)
        self.__lmodel = model.to(self.__device)
        self.__ldr_train = dataset[0]
        self.__ldr_test = dataset[1]
        
        self.__lr = self.args.lr
        self.__lr_decay = self.args.lr_decay
        self.__optimizer = self.__initialOptimizer(self.args.optimizer,self.__lmodel,self.__lr,self.args.momentum)          
        self.__scheduler = optim.lr_scheduler.StepLR(self.__optimizer, step_size=1, gamma=self.args.gamma)
        self.__affinity_clusters = [0]
        self.clabel = 0
        self.lgn = 0.0
        self.m = len(self.__ldr_train)/dataset[2]
        
    def __initialOptimizer(self,optimizer,model,lr,momentum) -> object:            
        if optimizer == "SGD":
            opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
        else:
            opt = optim.Adam(params=model.parameters(),lr=lr)
        return opt
        
    def preTrain(self) -> None:
        self.__lmodel.train()
        optimizer = optim.SGD(params=self.__lmodel.parameters(), lr=0.001, momentum=0.996)
        for _ in range(self.args.pre_epoch):
            for data, target in self.__ldr_train:
                data, target = Variable(data).to(self.__device), Variable(target).to(self.__device)
                optimizer.zero_grad()
                output = self.__lmodel(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
        print(f'client {self.__id} pre-train completed')

    def localTraining(self) -> None:
        self.__lmodel.train()
        epoch_loss = []
        if self.__delay is not None:
            round = max(int(self.args.local_ep*(1-self.__delay)),1)
        else:
            round = self.args.local_ep
        for _ in range(round):
            batch_loss = []
            for images, labels in self.__ldr_train:
                images, labels = Variable(images).to(self.__device), Variable(labels).to(self.__device)
                self.__optimizer.zero_grad()
                log_probs = self.__lmodel(images)
                loss = self.__lossfunc(log_probs, labels)
                loss.backward()
                self.__optimizer.step()
                batch_loss.append(loss.item())
            
            self.__scheduler.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
 
    def updateLocalModel(self, global_weights) -> None:

        upd_weights = [global_weights[label] for label in self.__affinity_clusters]
        if upd_weights == []: return
        upd_weight = copy.deepcopy(upd_weights[0])
        l = len(upd_weights)
        for key in upd_weight.keys():
            for i in range(1, l):
                upd_weight[key] += upd_weights[i][key]
            upd_weight[key]= torch.div(upd_weight[key], l)
        self.__lmodel.load_state_dict(upd_weight)

    def computeLocalGradientNorm(self) -> None:
        self.__lmodel.eval()
        grad_norm = 0.0
        with torch.no_grad():
            for param in self.__lmodel.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad.data, p=2).item() ** 2
        self.lgn = - self.__lr * grad_norm
        
        
    def evalLocalModel(self) -> float:
        test_loss = 0.0
        test_correct = 0.0
        self.__lmodel.eval()
        with torch.no_grad():  
            for data, label in self.__ldr_test:
                data, label = Variable(data).to(self._device), Variable(label).to(self._device)
                output = self.__lmodel(data) 
                test_loss += nn.CrossEntropyLoss()(output, label).item()
                test_correct += (torch.sum(torch.argmax(output,dim=1)==label)).item()
            test_correct /= len(self.__ldr_test.dataset)
        return test_correct
    
