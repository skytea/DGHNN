import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv
from torch_geometric.utils import add_self_loops
from collections import OrderedDict
import numpy as np


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 depths,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat = False,
                 dropout = 0.5,
                 leaky_relu_slope = 0.2,
    ):
        super(GCN, self).__init__()
        self.ml = nn.ModuleList()
        self.res = nn.ModuleList()
        in_nodes = in_features
        for index in range(len(n_hidden)):
            res_layers = nn.ModuleList()
            nodes = n_hidden[index]
            feature_in = in_nodes
            for layer in range(depths[index]):
                if n_heads == 0:
                    layer_temp = GCNConv(in_channels=feature_in, out_channels=nodes)
                else:
                    layer_temp = GATConv(in_channels=feature_in, out_channels=nodes, heads=n_heads, concat=concat, dropout=dropout)
                feature_in = nodes
                res_layers.append(layer_temp)

            if in_nodes == nodes:
                temp_res = nn.Identity()
            else:
                temp_res = nn.Linear(in_nodes, nodes)

            in_nodes = nodes
            self.ml.append(res_layers)
            self.res.append(temp_res)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout

    def forward(self, input_tensor, edge_index):
        x = input_tensor
        for index in range(len(self.ml)):
            x_res = x
            x_res = self.res[index](x_res)
            for layer in self.ml[index]:
                x = layer(x, edge_index)
                x = self.leaky_relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)
            x = x + x_res

        return x


class HyperGCN(nn.Module):
    def __init__(self,
                 in_features,
                 depths,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat = False,
                 dropout = 0.5,
                 leaky_relu_slope = 0.2,
    ):
        super(HyperGCN, self).__init__()
        self.ml = nn.ModuleList()
        self.res = nn.ModuleList()
        in_nodes = in_features
        for index in range(len(n_hidden)):
            res_layers = nn.ModuleList()
            nodes = n_hidden[index]
            feature_in = in_nodes
            for layer in range(depths[index]):
                layer_temp = HypergraphConv(in_channels=feature_in, out_channels=nodes)
                feature_in = nodes
                res_layers.append(layer_temp)

            if in_nodes == nodes:
                temp_res = nn.Identity()
            else:
                temp_res = nn.Linear(in_nodes, nodes)

            in_nodes = nodes
            self.ml.append(res_layers)
            self.res.append(temp_res)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout

    def forward(self, input_tensor, edge_index):
        x = input_tensor
        for index in range(len(self.ml)):
            x_res = x
            x_res = self.res[index](x_res)
            for layer in self.ml[index]:
                x = layer(x, edge_index)
                x = self.leaky_relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)
            x = x + x_res

        return x


class DGHNN(nn.Module):
    def __init__(self,
                 in_features,
                 depths,
                 n_hidden,
                 n_heads,
                 num_classes,
                 concat = False,
                 dropout = 0.5,
                 leaky_relu_slope = 0.2,
                 meta_x = None,
                 data = None,
                 node2idx = None,
    ):
        super(DGHNN, self).__init__()
        self.gat = GCN(in_features=in_features, depths=depths, n_hidden=n_hidden, n_heads=n_heads, num_classes=num_classes, concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope)
        self.hgat = HyperGCN(in_features=in_features, depths=depths, n_hidden=n_hidden, n_heads=n_heads, num_classes=num_classes, concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope)

        self.linear = nn.Linear(n_hidden[-1] * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout
        nn.init.xavier_normal_(self.linear.weight)

        x = data.x.float()
        self.nb_nodes = x.shape[0]
        node_names = np.concatenate(data.node_names, axis=0)
        meta_edge_index = [[], []]

        for i, node in enumerate(node_names):
            meta_edge_index[0].append(i)
            meta_edge_index[1].append(node2idx[tuple(node)] + x.shape[0])

        self.meta_edge_index = torch.tensor(meta_edge_index).cuda()
        self.meta_edge_index, _ = add_self_loops(self.meta_edge_index)
        self.meta_x = meta_x.cuda()

        self.meta_linear = nn.Linear(in_features, n_hidden[-1])
        self.meta_gnn = GCNConv(n_hidden[-1], n_hidden[-1])

    def forward(self, input_tensor, adj_mat, h_matrix):
        meta_x = self.leaky_relu(self.meta_linear(self.meta_x))

        number_nodes_of_x = input_tensor.shape[0]
        number_nodes_of_meta_x = meta_x.shape[0]

        x = self.gat(input_tensor, adj_mat)
        x = self.meta_gnn(torch.cat((x, meta_x), dim=0), self.meta_edge_index)
        x = self.leaky_relu(x)
        _, x = torch.split(x, [number_nodes_of_x, number_nodes_of_meta_x], dim=0)

        h_x = self.hgat(self.meta_x, h_matrix)

        x = torch.cat((x, h_x), dim=1)

        return x