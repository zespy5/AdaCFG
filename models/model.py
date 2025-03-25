from typing import Literal, Optional,List
import torch
import torch.nn as nn
from models.attn_module import *
       
  
class GuidanceModel(nn.Module):
    def __init__(
        self,
        init_g : Optional[float] = None,
        divide_out : float = 0.1,
        num_guidance_info : int = 1,
        linear_in_size: Optional[int] = None,
        num_mlp_layers: int = 2,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        **kwargs
    ):
        super().__init__()
        activates = {"gelu": nn.GELU(), "mish": nn.Mish(), "silu": nn.SiLU()}
        
        
        self.num_guidance_info = num_guidance_info
        assert self.num_guidance_info==1 or self.num_guidance_info==2, 'The number of guidance info must be 1 or 2'
        self.in_size = linear_in_size
        self.hidden_act = hidden_act
        self.num_mlp_layers = num_mlp_layers
        self.relu = nn.ReLU()
    
        
        self.init_G = 50.0 if init_g is None else init_g
        self.divide_out = divide_out

        self.MLP_modules = []
        self.activate = activates[hidden_act]
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size))
            self.MLP_modules.append(self.activate)
        
        self.MLP_modules.append(nn.LayerNorm(self.in_size))
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        self.layernorm = nn.LayerNorm(self.in_size)
        self.out = nn.Linear(self.in_size, self.num_guidance_info)
        
    def forward(self, x):

        out = self.MLP(x)
        out = self.layernorm(out)
        out = self.out(out)

        out *=self.divide_out
        out = self.relu(2*(torch.sigmoid(out)-0.5))
        out = out*self.init_G+1
        return out
        
    
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

    '''def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        for block in self.attnblocks:
            hidden_states = block(hidden_states)
        
        output = hidden_states.view(batch_size,-1)
        output = self.W(output)
        output = output*self.divide_out
        
        g_init = torch.sigmoid(output)*self.init_g +1
        
        return g_init '''  
     
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
    