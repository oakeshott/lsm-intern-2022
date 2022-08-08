import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, GCNConv

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, self.output_dim),
#             nn.Softmax(dim=1), # loss functionにsoftmaxが組み込まれているため不要
        )

    def forward(self, x):
        return self.mlp(x)

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = 128

        self.linear = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.output_dim),
                )

        self.gcn1 = GCNConv(self.input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.gcn1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear(global_add_pool(x, batch))

