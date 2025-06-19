#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import copy
import torch


MODEL_PATH = './models/cache/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

class ServiceProvider(object):
    
    def __init__(self,model,device,args) -> None:
        self.model = copy.deepcopy(model.to(device))
        self.device = device
        self.args = args
    
    def obscure_model(self, participant_ids):
        self.blind_factors = {}
        for cid in participant_ids:
            client_model = copy.deepcopy(torch.load(MODEL_PATH+'client_model_{}.pt'.format(cid)))
            for name, param in client_model.named_parameters():
                # 生成与权重形状相同的随机盲化因子
                blind_factor = torch.randn_like(param.data)
                # 保存盲化因子
                self.blind_factors[name] = blind_factor
                # 盲化权重
                param.data += blind_factor
    
    def unblind_model(self, model):
        obscured_weights = {name: param.data.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            # 去盲化
            param.data = obscured_weights[name] - self.blind_factors[name]

   