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
        
        
class IP2PGuidanceModel(nn.Module):
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
        out = torch.sigmoid(out)
        out = out*self.init_g+1
        return out
    
    def _forward2(self, x):
        
        for block in self.ff_block:
            x = block(x)

        out = self.W(x)
        out *= self.divide_out
        out_g = out[:,0]
        out_v = out[:,1]

        out_g = torch.sigmoid(out_g)
        out_g = out_g*self.init_g+1
        
        out_v = torch.exp(-out_v)
        return out_g.unsqueeze(1), out_v.unsqueeze(1)
    
    def forward(self, x):
        if self.num_guidance_info==1:
            return self._forward1(x)
        else:
            return self._forward2(x)
        