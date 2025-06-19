import numpy as np
import matplotlib.pyplot as plt

import copy
import torch
from torchvision import transforms, datasets
from sklearn.utils.extmath import randomized_svd

from utils.options import args_parser
from utils.data_loader import DatasetLoader


MODEL_PATH = './models/cache/'

# def Construct_Matrix_W(data,k=5):
#     rows = len(data) # 取出数据行数
#     W = np.zeros((rows,rows)) # 对矩阵进行初始化：初始化W为rows*rows的方阵
#     for i in range(rows): # 遍历行
#         for j in range(rows): # 遍历列
#             if(i!=j): # 计算不重复点的距离
#                 W[i][j] = gaussian_kernel(data[i],data[j]) # 调用函数计算距离
#         t = torch.argsort(torch.tensor(W[i,:])) # 对W中进行行排序，并提取对应索引
#         for x in range(rows-k): # 对W进行处理
#             W[i][t[x]] = 0
#     W = (W+W.T)/2 # 主要是想处理可能存在的复数的虚部，都变为实数
#     return W

# load model and get para
def load_clients(participants):
    model_states =[[] for _ in range(len(participants))]
    for p_id, p in enumerate(participants):
        grad_model = copy.deepcopy(p._model.state_dict())       
        for name in grad_model.keys():
            a = grad_model[name].view(-1).tolist()
            model_states[p_id].extend(a)
        
    return torch.tensor(model_states)

# get cluster number by using SVD
def compute_cluster_number(n_clients,clients):
    k = 1
    matrix = torch.transpose(clients.clone().detach(), 0, 1)
    matrix_np = matrix.cpu().numpy()

    _, S_np, _ = randomized_svd(matrix_np, n_components=n_clients, n_iter=10, random_state=None)
    S = torch.tensor(S_np).to(matrix.device)
    # _, s, _ = torch.svd_lowrank(a, q=n_clients)  
    # plt.pie(s, autopct='%1.1f%%')
    # plt.savefig("./logs/a_{}.jpg".format(epoch))
    # plt.show()
    
    for k in range(len(S)):  
        # if (torch.sum(s[0:k]) / torch.sum(s)).item() >= (torch.max(s).item() + float((np.ceil(epoch / 10)) / 10)):
        if torch.sum(S[0:k]) / torch.sum(S) >= 0.7:
            break
    print(k)


    return k

def compute_laplacian(S):
    # 计算标准化拉普拉斯矩阵
    D = torch.diag(S.sum(0))
    L = D - S
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(S.sum(0)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym


def visualize_soft_clusters(data, memberships, threshold=0.4):
    # 设置每个簇对应的颜色
    colors = np.array([[1, 0, 0], [0, 0, 1]])  # 红色和蓝色
    color_by_membership = np.dot(memberships, colors)  # 计算每个点的颜色
    
    plt.figure(figsize=(8, 6))
    # 绘制所有点
    plt.scatter(data[:, 0], data[:, 1], s=50, color=color_by_membership, alpha=0.5)

    # 对于每个点，如果满足条件，则显示隶属度
    for j in range(len(data)):
        if np.all(memberships[j, :] > threshold):  # 检查是否所有隶属度都超过阈值
            text = ', '.join(f'{m:.2f}' for m in memberships[j, :])
            plt.text(data[j, 0], data[j, 1], text, color='black', ha='center', va='center', fontsize=8)
    
    plt.title("Soft Clustering Visualization with Membership Display")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    args = args_parser()
    dataset_train = datasets.CIFAR10("data/dataset",
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))
                                    ]))

    labels = np.array(dataset_train.targets)
    dataloader = DatasetLoader(args=args)
    # 让每个client不同label的样本数量不同，以此做到Non-IID划分
    if args.iid == 'IID':
        client_idcs = dataloader.split_dataset_iid(dataset_train)
    else:
        client_idcs = dataloader.split_dirichlet_noniid_data(dataset_train)
    
    usr_distribution = [[] for _ in range(10)]
    label_distribution = np.zeros((args.num_user, 10)).astype(int)
    for c_id, idc in client_idcs.items():     
        for idx in idc:
            label_distribution[c_id][labels[idx]]+=1
            usr_distribution[labels[idx]].append(c_id)
            
    plt.figure(figsize=(10, 8))
    n, bins, patches = plt.hist(usr_distribution, stacked=True,
             bins=np.arange(-0.5, args.num_user + 1.5, 1),
             label=["Class {}".format(i) for i in range(10)],
             orientation='horizontal',
             rwidth=0.5)
    # for i in range(len(n)):
    #     plt.text(bins[i], n[i]*1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    plt.yticks(np.arange(args.num_user), ["{}"
               .format(c_id) for c_id in range(args.num_user)])
    plt.ylabel("Number of samples", size = 15)
    plt.xlabel("Client ID",size = 15)
    # plt.title('{} {}({})'.format(args.dataset,args.iid,args.beta),size = 15)
    plt.title('{} {}'.format(args.dataset,args.iid),size = 15)
    plt.legend(loc="upper right",ncol=2)
    # plt.savefig('result/{} {}({}).jpg'.format(args.dataset,args.iid,args.beta))
    plt.savefig('result/{} {}.jpg'.format(args.dataset,args.iid))
    plt.show()
    

    # n_client=args.num_user
    # clients = load_clients(n_clients=n_client)
    # clients = torch.tensor(clients).to(args.gpu)

    # distance = Distance_matrix(n_client, clients)      
    # clients = torch.tensor(distance)

    # k = SVD_k(n_client, clients)  
    # for index, knn_para in enumerate((5, 10, 20, 30)):
    #     clusters = spectral(clients, distance, k, knn_para)
        
    # plt.figure()
    # plt.scatter(label_distribution[:, 0], label_distribution[:, 1])
    # plt.title('Spectral clustering with {} clusters (KNN, neighbours={}).'.format(k, knn_para))
    # plt.show()