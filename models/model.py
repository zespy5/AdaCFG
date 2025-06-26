from typing import Literal, Optional,List
import torch
import torch.nn as nn
from models.attn_module import *
       
  
class GuidanceModel(nn.Module):
    def __init__(
        self,
        init_g : float = 50.0,
        divide_out : float=0.1,
        num_guidance_info : int=1,
        num_layers : int=2,
        hidden_dim : int=768,
        **kwargs,
    ):
        super().__init__()
        self.init_g = init_g
        assert num_guidance_info==1 or num_guidance_info==2, "num_guidance_info must be 1 or 2"
        self.num_guidance_info = num_guidance_info
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.divide_out = divide_out

        self.ff_block = nn.ModuleList(
            [FeedForward(hidden_dim, 2) for _ in range(num_layers)]
        )
        
        self.W = nn.Linear(self.hidden_dim, self.num_guidance_info)
        
    def _forward1(self, x):
        
        for block in self.ff_block:
            x = block(x)

        out = self.W(x)
        out *=self.divide_out
        out = self.relu(2*(torch.sigmoid(out)-0.5))
        out = out*self.init_g+1
        return out
    
    def _forward2(self, x):
        
        for block in self.ff_block:
            x = block(x)

        out = self.W(x)
        out *= self.divide_out
        out_g = out[:,0]
        out_v = out[:,1]

        out_g = self.relu(2*(torch.sigmoid(out_g)-0.5))
        out_g = out_g*self.init_g+1
        
        out_v = torch.exp(-out_v)
        return out_g.unsqueeze(1), out_v.unsqueeze(1)
    
    def forward(self, x):
        if self.num_guidance_info==1:
            return self._forward1(x)
        else:
            return self._forward2(x)
        
    
class AttentionModel(nn.Module):
    
    def __init__(self,
                 init_g : float,
                 divide_out : float,
                 num_guidance_info : int,
                 num_layers : int,
                 hidden_dim : int,
                 heads : int=8,
                 length : int=3,
                 **kwargs,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.init_g = init_g
        self.divide_out = divide_out
        self.num_guidance_info = num_guidance_info
        self.relu = nn.ReLU()
        
        self.attnblocks = nn.ModuleList(
            [AttnBlock(hidden_dim, heads) for _ in range(num_layers)]
        )
        
        self.W = nn.Linear(self.hidden_dim*length, self.num_guidance_info)

     
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        for block in self.attnblocks:
            hidden_states = block(hidden_states)
        
        output = hidden_states.view(batch_size,-1)
        output = self.W(output)
        output = output*self.divide_out
        out = self.relu(2*(torch.sigmoid(output)-0.5))
        g_init = out*self.init_g +1
        
        return g_init
        
    
class MultiConditionAttentionBLIPModel(nn.Module):
    
    def __init__(self,
                 init_blip_g : float,
                 init_g : float,
                 divide_out : float,
                 num_layers : int,
                 hidden_dim : int,
                 heads : int=8,
                 length : int=3,
                 **kwargs,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.init_g = init_g
        self.divide_out = divide_out
        self.init_blip_g = init_blip_g
        self.legth = length
        self.relu = nn.ReLU()
        
        self.attnblocks = nn.ModuleList(
            [AttnBlock(hidden_dim, heads) for _ in range(num_layers)]
        )
        
        self.blip_W = nn.Linear(self.hidden_dim*self.legth, 1)
        self.W = nn.Linear(self.hidden_dim*self.legth, 1)


        
    def forward(self, hidden_states):

        for block in self.attnblocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        
        blip_output = self.blip_W(hidden_states)
        init_blip_g = blip_output*self.divide_out
        
        blip_g_init = self.relu(2*(torch.sigmoid(init_blip_g)-0.5))*self.init_blip_g +1
        
        output = self.W(hidden_states)
        init_g = output*self.divide_out
        g_init = self.relu(2*(torch.sigmoid(init_g)-0.5))*self.init_g +1

        return blip_g_init, g_init
    