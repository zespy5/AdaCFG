import torch
import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    
    def __init__(self,
                 hidden_dim: int,
                 heads : int,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads


        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, 1e-6)
        
    def forward(self, hidden_states):
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        
        query = self.to_q(hidden_states)
        key   = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        
        head_dim = self.hidden_dim//self.heads
        
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1,2)
        key   = key.view(batch_size, -1, self.heads, head_dim).transpose(1,2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1,2)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        
        hidden_states = hidden_states.transpose(1,2).reshape(batch_size,-1, self.hidden_dim)
        
        output = self.layer_norm(self.to_out(hidden_states)+residual)
        
        return output
    
class FeedForward(nn.Module):
    
    def __init__(self,
                 hidden_dim:int,
                 ):
        super().__init__()
        
        self.W1 = nn.Linear(hidden_dim, 4*hidden_dim)
        self.W2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, 1e-6)
        self.activate = nn.GELU()
        
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.W2(self.activate(self.W1(hidden_states)))
        output = self.layer_norm(hidden_states+residual)
        
        return output
 