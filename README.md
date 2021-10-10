# Two Graph-based Solutions for Intrusion Detection
This repository contains the implementation of the modified Edge-based GraphSAGE ([E-GraphSAGE](https://arxiv.org/abs/2103.16329)) and Edge-based Residual Graph Attention Network (E-ResGAT) as well as their original versions. They are designed to solve intrusion detecton tasks in a graph-based manner.

# Background
Cybersecurity has drawn growing attention in contemporary society, due to the increasingly sophisticated cyber criminals. There still remain many challenges that are not fully solved in this research area. Recently, to model the network topology, graph-based deep learning has achieved state-of-the-art performance in cybersecurity tasks. This paper presents two graph-based solutions for intrusion detection, the modified E-GrphSAGE and E-ResGAT algorithms. They are modified from the established GraphSAGE and GAT models, respectively, to tailor intrusion detection network. ???However, such task usually suffers from extreme class imbalance between normal and malicious traffics. Our proposed models add residual connections in hope of retaining the original information and performing better on the minority classes. An extensive experimental evaluation based on four recent intrusion detection datasets shows the excellent performance of our models in most cases, especially when predicting minority classes.

# Demo
To replicate results in the paper, call `fit_mdoel.py` with appropriate arguments. The datasets CES-CIC, Darknet, ToN-IoT and UNSW-NB15 are supported here. For example, to replicate the E-ResGAT multi-classification model on UNSW-NB15 dataset, call

    
    `python fit_model.py --alg="gat" --dataset="UNSW-NB15" --binary=False --residual=True`
    
Besides, the original E-GraphSAGE and GAT are also implemented. One can simply run the original version by including the argument `--residual = False`.

## Installation
This implementation requires Python 3.X. See `requirements.txt` for a list installed packages and their versions. The main packages are:
