import numpy as np
import torch
import dgl
import util.util as nutil


# check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_node_feat = 0


def read_dataset(ds_name, num_classes):
    samples = []
    data_adj = np.loadtxt('data/' + ds_name + '/' + ds_name + '_A.txt', delimiter=',').astype(int) - 1
    data_node_att = nutil.zscore(np.loadtxt('data/' + ds_name + '/' + ds_name + '_node_attributes.txt', delimiter=','))
    data_graph_indicator = np.loadtxt('data/' + ds_name + '/' + ds_name + '_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_label = np.loadtxt('data/' + ds_name + '/' + ds_name + '_graph_labels.txt', delimiter=',').astype(int)

    graph = dgl.DGLGraph()
    graph.add_nodes(len(data_graph_indicator))
    graph.ndata['feat'] = torch.tensor(data_node_att, dtype=torch.float32).to(device)
    graph.add_edges(list(data_adj[:, 0]), list(data_adj[:, 1]))

    for i in range(0, data_graph_label.shape[0]):
        nodes = np.where(data_graph_indicator == i + 1)[0]
        sub_graph = graph.subgraph(nodes)
        sub_graph.copy_from_parent()
        samples.append((sub_graph, data_graph_label[i] - 1))

    global dim_node_feat
    dim_node_feat = data_node_att.shape[1]

    return samples
