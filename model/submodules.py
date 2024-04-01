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
                 data, 
                 type: str="GCN"):
    
        super(GNN, self).__init__()
    
        # Stores size of each layer in a list
        layer_sizes = [node_feature_size] + [hidden_dim] * (num_layers) + [output_embedding_size]
    
        # Set the type of GNN propagation mechanism to be used
        
        assert type in ["GCN", "GAT"], "Invalid GNN prop layer selected"
        
        if type == "GCN":
            prop_layer = GCNConv
        
        elif gnn == "GAT":
            prop_layer = GATConv
            
        # Layers are GCN / GAT layers
        self.layers = nn.ModuleList([prop_layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)])
        
        # Activation function
        self.activation = nn.ReLU()
  
        self.data = data

        
    def forward(self):
        """
        Performs forward pass through GCN / GAT.
        
        Return
        ------
        x : _
            Modified data (now passed through network).
        """
        x = self.data.x
        index = self.data.edge_index
        
        for i, aggregator in enumerate(self.layers):
            # Aggregate for each layer
            x = aggregator(x, index)
            
            # Apply ReLu (activation) after every layer
            if not i == len(self.layers) - 1:
                x = self.activation(x)
                
        return x # Return passed forward data
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(MLP, self).__init__()
         
        # Stores size of each layer in a list
        layer_sizes = [input_dim] + [hidden_dim]*(num_layers) + [output_dim]
        
        # Standard fully connected linear layers
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])
        
        # Activation function is set to ReLU
        self.activation = nn.ReLU()


    def forward(self, x):
        """
        Performs forward pass through fully connected linear layers.
        
        Parameters
        ----------
        x : _
            Data passed into the network.
            
        Returns
        ------
        x : _ 
            Data transformed through the network.
        """
        
        for i, linear_layer in enumerate(self.layers):
            x = linear_layer(x) # Apply the linear transformation
            
            # Apply ReLU (activation) after each layer
            if not i == len(self.layers) - 1:
                x = self.activation(x)
            
        return x # Return passed forward data