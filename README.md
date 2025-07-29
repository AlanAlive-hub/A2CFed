# AAAI2026 Submission 

《Less is more: Clustering with Adaptive Probability for Heterogeneous Federated Learning》

## 介绍

Federated learning (FL) is a distributed machine learning paradigm that enables multiple clients to collaboratively train a model without sharing their private dataset. However, statistical heterogeneity is a crucial challenge that significantly limits training efficiency and model performance. Existing FL methods often utilize clustering to group clients with isomorphic-like data distribution. However, clustering selection and adaptation introduce challenges that affect overall efficiency, while intra-cluster data exchange poses privacy risks. To address these issues, this article introduces a novel method, called CAPFed, which performs clustering in adaptive probability while obliviously completing clustering. The key idea is to conduct clustering with a higher probability in the most critical periods, i.e., those with the highest information entropy, where we evaluate the local training performance for the adaptive implementation of clustering. Furthermore, CAPFed adopts a Shuffle-based identity anonymization that provides adequate and efficient protection against the privacy leakage of similarities between clients during the clustering process. Due to its independence from inter-client optimization approaches, it enables seamless integration with other baselines for improved performance. Experimental results show that our method achieves an improvement in ∼7× in terms of computational efficiency while obtaining the optimal model performance.

