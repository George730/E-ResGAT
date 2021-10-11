# Graph-based Solutions with residuals for Intrusion Detection
This repository contains the implementation of the modified Edge-based GraphSAGE ([E-GraphSAGE](https://arxiv.org/abs/2103.16329)) and Edge-based Residual Graph Attention Network (E-ResGAT) as well as their original versions. They are designed to solve intrusion detecton tasks in a graph-based manner.

# Background
Cybersecurity has drawn growing attention in contemporary society, due to the increasingly sophisticated cyber threats. Many challenges still remain to solve and these problems are an active research. Namely, for intrusion detection, new algorithms that are more robust, effective and able to use more information available are needed. Moreover, the intrusion detection task usually suffers from extreme class imbalance between normal and malicious traffic which is still an unsolved issue. Recently, graph-based deep learning has achieved state-of-the-art performance to model the network topology in cybersecurity tasks. However, much remain to be researched as this is an active research area, where only a few works exist using GNNs to tackle the intrusion detection problem. Moreover, other promising avenues such as the use of the attention mechanism are still under-explored. This paper presents two novel graph-based solutions for intrusion detection, the modified E-GraphSAGE and E-ResGAT algorithms. The first is based on the established GraphSAGE while the second relies on the graph attention networks (GATs) to tackle the network intrusion detection problem. The key idea of these solutions is to integrate residual learning into the GNN leveraging the graph information available. Our proposed models add residual connections as a strategy to deal with the high class imbalance, aiming at retaining the original information and performing better on the minority classes. An extensive experimental evaluation based on four recent intrusion detection datasets shows the excellent performance of our models, especially when predicting minority classes.

# Demo
To replicate results in the paper, call `fit_mdoel.py` with appropriate arguments. The datasets CES-CIC, Darknet, ToN-IoT and UNSW-NB15 are supported here. For example, to replicate the E-ResGAT multi-classification model on UNSW-NB15 dataset, call

    
    python fit_model.py --alg="gat" --dataset="UNSW-NB15" --binary=False --residual=True
    
Besides, the original E-GraphSAGE and GAT are also implemented. One can simply run the original version by including the argument `--residual = False`.

### Installation
This implementation requires Python 3.X. See `requirements.txt` for a list installed packages and their versions. The main packages are:

    numpy
    pytorch
    scikit-learn
