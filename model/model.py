import torch
import torch.nn as nn

from .submodules import GNN, MLP # GNN used in the complete model

class Model(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 graph=None, 
                 device: str="cpu", 
                 args=None, 
                 embeddings=None, 
                 mapping=None, 
                 traffic_matrix=None):

        super(Model, self).__init__()
        
        self.args = args
        
        if embeddings is None: # If embeddings are not provided
            # Randomly initialize them. Dimension: (num_nodes) x (embedding_size)
            self.embeddings = nn.Embedding(num_nodes, args.embedding_size)
            self.embeddings = nn.Embedding.from_pretrained(self.embeddings.weight, freeze=(not args.trainable_embeddings))

        else:
            # Create embedding layer using values passed into function
            self.embeddings = nn.Embedding.from_pretrained(embeddings.float(), freeze=(not args.trainable_embeddings))
            
        # Include gnn inside complete model, if specified
        if (args.gnn is not None):
            print(f"Graph Neural Network used: {args.gnn}")
            
            # Feature size = embeddings dimension
            # Can't be assigned directly to `ąrgs.embedding size`, for pretrained loading may not be of same dimension
            node_feature_size_for_gnn = self.embeddings.weight.shape[1]
            
            self.GNN = GNN(
                            node_feature_size=node_feature_size_for_gnn,
                            output_embedding_size=args.embedding_size, 
                            num_layers=args.gnn_layers, 
                            hidden_dim=args.hidden_size, 
                            graph=graph,
                            gnn_type=args.gnn
                          )

        input_size = 6 * args.embedding_size
        self.mapping = mapping
        self.device  = device  # Device to run model on
        
        
        # TODO Comments
        if args.traffic:         
            input_size = 8 * args.embedding_size # one for traffic
            
            self.traffic_matrix         = nn.Embedding.from_pretrained(traffic_matrix, freeze=True)	
            self.traffic_linear_initial = nn.Linear(self.traffic_matrix.weight.shape[1], 2 * args.embedding_size)	
        
        # Use multihead attention, as described in [1]
        if args.attention:
            self.self_attention = nn.MultiheadAttention(2 * self.embeddings.weight.shape[1], args.num_heads)
            
        # Standard Multi-layer perceptron architecture. Outputs a single real value.
        self.confidence_model = MLP(
                                     input_dim=input_size, 
                                     output_dim=1, 
                                     num_layers=args.num_layers, 
                                     hidden_dim=args.hidden_size
                                   ) 

    def forward(self, source, dest, nbr, traffic=None):
        """
        Performs a forward pass.
        """
        device = self.device
        edge_to_node_mapping = self.mapping
  
        source_left  = torch.LongTensor([edge_to_node_mapping[x][0] for x in source]).to(device)
        source_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in source]).to(device)
        
        nbr_left  = torch.LongTensor([edge_to_node_mapping[x][0] for x in nbr]).to(device)
        nbr_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in nbr]).to(device)
        
        dest_left  = torch.LongTensor([edge_to_node_mapping[x][0] for x in dest]).to(device)
        dest_right = torch.LongTensor([edge_to_node_mapping[x][1] for x in dest]).to(device)
        
        
        if (self.args.gnn is not None):
            # Data used in GNN are embeddings
            self.GNN.data.x = self.embeddings.weight
            embeddings = self.GNN() # forwward() method TODO?
        
        else:
            embeddings = self.embeddings.weight # Embeddigs not modified in place of GNN
  
        # Adding an extra row on top
        embeddings = torch.cat((torch.zeros(embeddings.shape[1]).reshape(1, -1).to(self.device), embeddings), dim=0)	
        source_vec = torch.cat((embeddings[1 + source_left], embeddings[1 + source_right]), dim=1)
        dest_vec   = torch.cat((embeddings[1 + dest_left],   embeddings[1 + dest_right]),   dim=1)
        nbr_vec    = torch.cat((embeddings[1 + nbr_left],    embeddings[1 + nbr_right]),    dim=1)
        
        if self.args.traffic:
            traffic_vec             = self.traffic_matrix(torch.LongTensor(traffic).to(device))
            traffic_vec_matched_dim = self.traffic_linear_initial(traffic_vec)

        if not self.args.attention:
            if self.args.traffic:
                x = torch.cat((source_vec, nbr_vec, dest_vec, traffic_vec_matched_dim), 1)
        
            else:
                x = torch.cat((source_vec, nbr_vec, dest_vec), 1)
        
        else:
            if self.args.traffic:
                q = torch.stack((source_vec, nbr_vec, dest_vec, traffic_vec_matched_dim))
            else:
                q = torch.stack((source_vec, nbr_vec, dest_vec))

            z, _ = self.self_attention(q, q, q)
            x = z.transpose(0,1).reshape(source_vec.shape[0],-1)

        y = self.confidence_model(x)
        y[nbr_left == -1] = -100
        
        return y