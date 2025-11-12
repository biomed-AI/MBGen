import torch
import torch.nn as nn
import torch.nn.functional as F

from src.many_body.layers import EN_Attention, FFN, DropPath
from src.many_body.many_body_attn import TripletAttention


class MBTransformerLayer(nn.Module):
    def __init__(
        self,
        node_width,
        edge_width,
        global_width,
        num_heads=8,
        ffn_multiplier=4,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        
        # Node-Edge attention
        self.attn = EN_Attention(
            node_width=node_width,
            edge_width=edge_width,
            num_heads=num_heads,
            source_dropout=attention_dropout,
            scale_degree=True,
            edge_update=True,
        )
        
        # Many-body edge interaction
        self.triplet = TripletAttention(
            edge_width=edge_width,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )
        
        # Global feature processing
        self.global_node_attn = nn.Linear(global_width, node_width)
        self.global_edge_attn = nn.Linear(global_width, edge_width)
        
        # Feed-forward networks
        self.ffn_h = FFN(
            width=node_width,
            multiplier=ffn_multiplier,
            act_dropout=dropout,
            activation='gelu',
        )
        
        self.ffn_e = FFN(
            width=edge_width,
            multiplier=ffn_multiplier,
            act_dropout=dropout,
            activation='gelu',
        )
        
        # Global feature update
        self.node_to_global = nn.Linear(node_width, global_width)
        self.edge_to_global = nn.Linear(edge_width, global_width)
        self.global_ffn = FFN(
            width=global_width,
            multiplier=ffn_multiplier,
            act_dropout=dropout,
            activation='gelu',
        )
        
        # Layer norms
        self.norm_h1 = nn.LayerNorm(node_width)
        self.norm_e1 = nn.LayerNorm(edge_width)
        self.norm_h2 = nn.LayerNorm(node_width)
        self.norm_e2 = nn.LayerNorm(edge_width)
        self.norm_g = nn.LayerNorm(global_width)
        
        # Droppath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, h, e, g, mask):

        batch_size, num_nodes = h.shape[0], h.shape[1]
        
        attn_mask = mask

        if mask.dim() == 3:
            tri_mask = mask.unsqueeze(-1)
        else:
            tri_mask = mask
        
        # Apply global features to nodes and edges 
        g_h = self.global_node_attn(g).unsqueeze(1)  # [B, 1, node_width]
        g_e = self.global_edge_attn(g).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, edge_width]
        
        # Node-Edge attention with global conditioning
        h_res, e_res = self.attn(h, e, attn_mask)
        h = h + self.drop_path(h_res) * (1 + g_h)
        e = e + self.drop_path(e_res) * (1 + g_e)
        h = self.norm_h1(h)
        e = self.norm_e1(e)
        
        # Many-body edge interaction
        e_res = self.triplet(e, tri_mask)
        e = e + self.drop_path(e_res) * (1 + g_e)
        e = self.norm_e2(e)
        
        # Feed-forward networks
        h = h + self.drop_path(self.ffn_h(h))
        e = e + self.drop_path(self.ffn_e(e))
        h = self.norm_h2(h)
        e = self.norm_e2(e)
        
        # Update global features
        g_update = torch.mean(self.node_to_global(h), dim=1) + \
                torch.mean(torch.mean(self.edge_to_global(e), dim=1), dim=1)
        g = g + self.drop_path(g_update)
        g = self.norm_g(g)
        g = g + self.drop_path(self.global_ffn(g))
        g = self.norm_g(g)
        
        return h, e, g


class MBTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dims,
        hidden_dims,
        hidden_mlp_dims,
        output_dims,
        num_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.0,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
        **kwargs
    ):
        super().__init__()
        
        # Input dimensions
        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        
        # Hidden dimensions
        self.hidden_dim_X = hidden_dims['dx']
        self.hidden_dim_E = hidden_dims['de']
        self.hidden_dim_y = hidden_dims['dy']
        
        # Output dimensions
        self.output_dim_X = output_dims['X']
        self.output_dim_E = output_dims['E']
        self.output_dim_y = output_dims['y']
        
        # Input projections
        self.node_encoder = nn.Sequential(
            nn.Linear(self.Xdim, hidden_mlp_dims['X']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], self.hidden_dim_X)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.Edim, hidden_mlp_dims['E']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], self.hidden_dim_E)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.ydim, hidden_mlp_dims['y']),
            act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], self.hidden_dim_y)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MBTransformerLayer(
                node_width=self.hidden_dim_X,
                edge_width=self.hidden_dim_E,
                global_width=self.hidden_dim_y,
                num_heads=num_heads,
                ffn_multiplier=4,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path * (i / (n_layers - 1)) if n_layers > 1 else 0,
            )
            for i in range(n_layers)
        ])
        
        # Output projections
        self.node_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim_X, hidden_mlp_dims['X']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], self.output_dim_X)
        )
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim_E, hidden_mlp_dims['E']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['E'], self.output_dim_E)
        )
        
        self.global_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim_y, hidden_mlp_dims['y']),
            act_fn_out,
            nn.Linear(hidden_mlp_dims['y'], self.output_dim_y)
        )
    
    def forward(self, X, E, y, node_mask):
        """
        X: Node features [batch_size, num_nodes, node_feature_dim]
        E: Edge features [batch_size, num_nodes, num_nodes, edge_feature_dim]
        y: Global features [batch_size, global_feature_dim]
        node_mask: Node mask [batch_size, num_nodes]
        """
        # Create attention mask 
        mask = node_mask.float()
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B, N, N]
        mask = torch.where(mask > 0, 0.0, torch.finfo(X.dtype).min)
        
        # Encode inputs
        h = self.node_encoder(X)
        e = self.edge_encoder(E)
        g = self.global_encoder(y)
        
        # Apply transformer layers
        for layer in self.layers:
            h, e, g = layer(h, e, g, mask)
        
        # Decode outputs
        X_out = self.node_decoder(h)
        E_out = self.edge_decoder(e)
        y_out = self.global_decoder(g)
        
        # Apply node mask
        X_out = X_out * node_mask.unsqueeze(-1)
        
        # Ensure edge symmetry
        E_out = (E_out + E_out.transpose(1, 2)) / 2
        
        # Apply edge mask (based on node mask)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        E_out = E_out * edge_mask.unsqueeze(-1)
        
        from src import utils
        return utils.PlaceHolder(X=X_out, E=E_out, y=y_out)