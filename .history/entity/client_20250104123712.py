#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
import torch
import itertools
from torch import nn, optim
from torch.autograd import Variable


MODEL_PATH = './models/cache/'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


class Client(object):
    def __init__(self, id, model, device, dataset, args):
        self._id = id
        self._device = device
        self._delay = 0.0
        self._args = args
        self._loss_func = nn.CrossEntropyLoss().to(self._device)
        self._model = model.to(self._device)
        self._ldr_train = dataset[0]
        self._ldr_test = dataset[1]
        self._lr = args.lr
        self._lr_decay = args.lr_decay
        self._optimizer = self.__initialOptimizer(args.optimizer,self._model,self._lr,args.momentum)          
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=5, gamma=self._args.gamma)
        self._label = 0
        self._affinity_clusters = [0]
        
    def __initialOptimizer(self,optimizer,model,lr,momentum):            
        if optimizer == "SGD":
            opt = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
        else:
            opt = optim.Adam(params=model.parameters(),lr=lr)
        return opt
        
    def pre_train(self):
        self._model.train()
        optimizer = optim.SGD(params=self._model.parameters(), lr=0.001, momentum=0.996)
        for _ in range(self._args.pre_epoch):
            for data, target in self._ldr_train:
                data, target = Variable(data).to(self._device), Variable(target).to(self._device)
                optimizer.zero_grad()
                output = self._model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

        print(f'client {self._id} pre-train completed')

    def trainLocalModel(self):
        self._model.train()
        epoch_loss = []
        if self._delay is not None:
            round = max(int(self._args.local_ep*(1-self._delay)),1)
        else:
            round = self._args.local_ep
        for _ in range(round):
            batch_loss = []
            for images, labels in self._ldr_train:
                images, labels = Variable(images).to(self._device), Variable(labels).to(self._device)
                self._optimizer.zero_grad()
                log_probs = self._model(images)
                loss = self._loss_func(log_probs, labels)
                loss.backward()
                self._optimizer.step()
                batch_loss.append(loss.item())
            
            self._scheduler.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
 
    def updateLocalModel(self, global_weights):

        upd_weights = [global_weights[label] for label in self._affinity_clusters]
        if upd_weights == []: return
        upd_weight = copy.deepcopy(upd_weights[0])
        l = len(upd_weights)
        for key in upd_weight.keys():
            for i in range(1, l):
                upd_weight[key] += upd_weights[i][key]
            upd_weight[key]= torch.div(upd_weight[key], l)
        self._model.load_state_dict(upd_weight)
        
        
    def evalLocalModel(self):
        test_loss = 0.0
        test_correct = 0.0
        self._model.eval()

        with torch.no_grad():  
            for data, label in self._ldr_test:
                data, label = Variable(data).to(self._device), Variable(label).to(self._device)
                output = self._model(data) 
                test_loss += nn.CrossEntropyLoss()(output, label).item()
                test_correct += (torch.sum(torch.argmax(output,dim=1)==label)).item()
            test_correct /= len(self._ldr_test.dataset)

        return test_correct
    
