import numpy as np
import matplotlib.pyplot as plt

import copy
import torch
from torchvision import transforms, datasets
from sklearn.utils.extmath import randomized_svd

from utils.options import args_parser
from utils.data_loader import DatasetLoader


MODEL_PATH = './models/cache/'

# load model and get para
def syn_model(models):
    model_states =[[] for _ in range(len(models))]
    for idx, m in enumerate(models):
        grad_model = copy.deepcopy(m.state_dict())       
        for name in grad_model.keys():
            a = grad_model[name].view(-1).tolist()
            model_states[idx].extend(a)
        
    return torch.tensor(model_states)

# get cluster number by using SVD
def compute_cluster_number(n_clients,clients):
    k = 1
    matrix = torch.transpose(clients.clone().detach(), 0, 1)
    matrix_np = matrix.cpu().numpy()

    _, S_np, _ = randomized_svd(matrix_np, n_components=n_clients, n_iter=10, random_state=None)
    S = torch.tensor(S_np).to(matrix.device)
    for k in range(len(S)):  
        if torch.sum(S[0:k]) / torch.sum(S) >= 0.7:
            break
    return k

def compute_laplacian(S):
    # Compute the Laplacian matrix from the similarity matrix S
    D = torch.diag(S.sum(0))
    L = D - S
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(S.sum(0)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym


def visualize_soft_clusters(data, memberships, threshold=0.4):
    # Visualize soft clustering results with membership values
    colors = np.array([[1, 0, 0], [0, 0, 1]])
    color_by_membership = np.dot(memberships, colors)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=50, color=color_by_membership, alpha=0.5)

    for j in range(len(data)):
        if np.all(memberships[j, :] > threshold):
            text = ', '.join(f'{m:.2f}' for m in memberships[j, :])
            plt.text(data[j, 0], data[j, 1], text, color='black', ha='center', va='center', fontsize=8)
    
    plt.title("Soft Clustering Visualization with Membership Display")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()