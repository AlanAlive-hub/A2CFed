# ICDE2026 Submission 

《$A^2$CFed: Anonymous Adaptive Clustering for Federated Learning with Statistic Heterogeneity》

## 介绍

Federated learning (FL) is a distributed machine learning paradigm that enables multiple clients to train a model collaboratively without exposing their local data. Among FL schemes, clustering is an effective technique to address the issue of heterogeneity (i.e., differences in data distribution and computational ability affect training performance and effectiveness) by grouping participants with similar computational resources or data distribution into clusters. However, intra-cluster data exchange poses privacy risks, while clustering selection and adaptation introduce challenges that may affect overall performance. To address these challenges, this paper introduces anonymous adaptive clustering, a novel approach that simultaneously enhances privacy protection and improves training efficiency. Specifically, an oblivious shuffle-based anonymization method is designed to protect user identities and prevent the aggregation server from inferring similarities through clustering. Additionally, to improve performance, we propose an adaptive frequency decay strategy based on the Crucial Learning Period that leverages variability in clustering probabilities to optimize training dynamics. Experimental results show that our method achieves an improvement in ∼ 7× in terms of performance while maintaining high privacy.

