#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epoch', default=100, type=int, help="rounds of training")
    parser.add_argument('--pre_epochs',default=1, type=int, help="rounds of pre-training")
    parser.add_argument('--early_stop', type=bool, help='early stopping')
    parser.add_argument('--stopping_round', type=int , help='rounds of early stopping')
    parser.add_argument('--num_user', default= 50, type=int, help="number of user")
    parser.add_argument('--frac', default=0.9, type=float, help="the fraction of clients: C")
    parser.add_argument('--local_ep', default=5, type=int, help="the number of local epochs: E")
    parser.add_argument('--local_bs', default=10, type=int, help="local batch size: B")
    parser.add_argument('--bs', default=128, type=int, help="test batch size")


    # model arguments
    parser.add_argument('--model', type=str, choices=["CNN","MCLR","RNN","ResNet18"], help='model name')
    parser.add_argument('--lr', default=0.001, type=float,help="learning rate")
    parser.add_argument('--lr_decay', default=0.995, type=float, help="learning rate decay each round")
    parser.add_argument('--momentum', default=0.997, type=float, help="SGD momentum")
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--gamma', default=0.99, type=float, choices=[1, 0.98, 0.99])
    
    # crypto arguments
    parser.add_argument('--secparam', default=1024,type=int, help="BCP security para")
    parser.add_argument('--t',default=0.8, type=float, help="threshold of signature scheme")

    # other arguments
    parser.add_argument('--dataset', default="MNIST", type=str, help="name of dataset", choices=["MNIST", "CIFAR10", "EMNIST","FMNIST"])
    parser.add_argument('--iid', default="Dir", type=str, help='whether i.i.d or not', choices=["IID", "nonIID", "Dir"])
    parser.add_argument('--beta', default=0.8, type=float, help='Dirchlet param')
    parser.add_argument('--delta', default=5, type=int, help='cluster frequence')
    parser.add_argument('--c', default=2, type=int, help='cluster number')      
    parser.add_argument('--split',  default="Dir", type=str, help="train-test split type, user or sample")
    parser.add_argument('--gpu', default=0,type=int, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    return args
