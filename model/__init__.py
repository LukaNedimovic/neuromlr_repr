import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNN(torch.nn.Module):
    def __init__(self, 
                node_feature_size, 
                 output_embedding_size, 
                 num_layers, 
                 hidden_dim, 
                 graph, 
                 gnn: str="GCN"):
    
        super(GNN, self).__init__()
    
        # Stores layer sizes in a list
        layer_sizes = [node_feature_size] + [hidden_dim] * (num_layers) + [output_embedding_size]
    
        # Set the type of GNN propagation mechanism to be used
        
        assert gnn in ["GCN", "GAT"], "Invalid GNN prop layer selected"
        
        if gnn == "GCN":
            prop_layer = GCNConv
        
        elif gnn == "GAT":
            prop_layer = GATConv
            
        self.layers = nn.ModuleList([prop_layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
        self.activation = nn.ReLU()
  
        self.data = graph
