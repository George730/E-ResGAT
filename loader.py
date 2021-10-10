import numpy as np
import torch
import torch.nn as nn
import pickle
from collections import defaultdict
from models import (MeanAggregator, Encoder)


def load_sage(path, binary):
    # nodes
    nodes = np.load(path+"nodes.npy", allow_pickle=True)
    num_nodes = len(nodes)

    # features node_feat: all one; edge_feat: scaled
    node_feat = np.ones((num_nodes, 64))
    node_features = nn.Embedding(node_feat.shape[0], node_feat.shape[1])
    node_features.weight = nn.Parameter(torch.FloatTensor(node_feat), requires_grad=False)
    edge_feat = np.load(path+"edge_feat_scaled.npy")  # (n,f)
    edge_features = nn.Embedding(edge_feat.shape[0], edge_feat.shape[1])
    edge_features.weight = nn.Parameter(torch.FloatTensor(edge_feat), requires_grad=False)

    # label
    if binary:
        label = np.load(path+"label_bi.npy", allow_pickle=True)  # (n,1)
    else:
        label = np.load(path+"label_mul.npy", allow_pickle=True)

    # mapping function from node ip to node id
    node_map = {}
    for i, node in enumerate(nodes):
        node_map[node] = i

    # adjacency adj: edge -> (node1, node2); adj_lists: {node: edge1, ..., edgen}
    adj = np.load(path+"adj.npy", allow_pickle=True)
    adj_lists = defaultdict(set)
    for i, line in enumerate(adj):
        node1 = node_map[line[0]]
        node2 = node_map[line[1]]
        adj_lists[node1].add(i)  # mutual neighbor
        adj_lists[node2].add(i)

    # Define two layer aggregators and encoders
    agg1 = MeanAggregator(edge_features, gcn=False, cuda=False)
    enc1 = Encoder(node_features, edge_feat.shape[1], 64, adj_lists,
                   agg1, num_sample=8, gcn=True, cuda=False)
    agg2 = MeanAggregator(edge_features, gcn=False, cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), edge_feat.shape[1], 64,
                   adj_lists, agg2, num_sample=8, base_model=enc1, gcn=True, cuda=False)

    return enc2, edge_feat, label, node_map, adj


def load_gat(path, device, binary):
    # feature
    edge_feat = np.load(path + "edge_feat_scaled.npy")  # （n,f)
    edge_feat = torch.tensor(edge_feat, dtype=torch.float, device=device)

    # label
    if binary:
        label = np.load(path + "label_bi.npy", allow_pickle=True)  # (n,1)
    else:
        label = np.load(path+"label_mul.npy", allow_pickle=True)
    label = torch.tensor(label, dtype=torch.long, device=device)  # Cross entropy expects a long int

    # adjacency
    adj = np.load(path + "adj_random.npy", allow_pickle=True)
    with open(path + 'adj_random_list.dict', 'rb') as file:
        adj_lists = pickle.load(file)

    # configuration
    config = {
        "num_of_layers": 3,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [6, 6, 6],
        "num_features_per_layer": [edge_feat.shape[1], 8, 8, 8],
        "num_identity_feats": 8,
        "add_skip_connection": False,
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.2  # result is sensitive to dropout
    }

    return edge_feat, label, adj, adj_lists, config
