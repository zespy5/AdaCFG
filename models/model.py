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
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        **kwargs
    ):
        super().__init__()
        activates = {"gelu": nn.GELU(), "mish": nn.Mish(), "silu": nn.SiLU()}
        
        
        self.num_guidance_info = num_guidance_info
        assert self.num_guidance_info==1 or self.num_guidance_info==2, 'The number of guidance info must be 1 or 2'
        self.in_size = hidden_dim
        self.hidden_act = hidden_act
        self.num_mlp_layers = num_layers
    
        
        self.init_G = 50.0 if init_g is None else init_g
        self.divide_out = divide_out

        self.MLP_modules = []
        self.activate = activates[hidden_act]
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size))
            self.MLP_modules.append(self.activate)
            
        self.MLP_modules.append(nn.BatchNorm1d(self.in_size))
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        self.batchnorm = nn.BatchNorm1d(self.in_size)
        self.out = nn.Linear(self.in_size, self.num_guidance_info)
        
    def forward(self, x):

        out = self.MLP(x)
        out = self.batchnorm(out)
        out = self.out(out)
        
        if self.num_guidance_info==1:
            out *=self.divide_out
            out = torch.sigmoid(out/self.init_G)*self.init_G+1
            return out
        
        elif self.num_guidance_info==2:
            g_init = out[:,0]*self.divide_out
            alpha = out[:,1]
            
            g_init = torch.sigmoid(g_init/self.init_G)*self.init_G+1
            alpha = torch.sigmoid(alpha)

            return g_init, alpha
            
  
class GuidanceModel2(nn.Module):
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
        
        self.MLP_modules.append(nn.BatchNorm1d(self.in_size))
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)
        self.batchnorm = nn.BatchNorm1d(self.in_size)
        self.out = nn.Linear(self.in_size, self.num_guidance_info)
        
    def forward(self, x):

        out = self.MLP(x)
        out = self.batchnorm(out)
        out = self.out(out)
        
        if self.num_guidance_info==1:
            out *=self.divide_out
            out = self.relu(2*(torch.sigmoid(out/self.init_G)-0.5))
            out = out*self.init_G+1
            return out
        
        elif self.num_guidance_info==2:
            g_init = out[:,0]*self.divide_out
            alpha = out[:,1]
            
            g_init = self.relu(2*(torch.sigmoid(g_init/self.init_G)-0.5))
            g_init = g_init*self.init_G+1
            alpha = torch.sigmoid(alpha)

            return g_init, alpha
    
class AttentionModel(nn.Module):
    
    def __init__(self,
                 init_g : float,
                 divide_out : float,
                 num_guidance_info : int,
                 num_layers : int,
                 hidden_dim : int,
                 heads : int=8,
                 **kwargs,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.init_g = init_g
        self.divide_out = divide_out
        self.num_guidance_info = num_guidance_info
        
        
        self.attnblocks = nn.ModuleList(
            [AttnBlock(hidden_dim, heads) for _ in range(num_layers)]
        )
        
        self.W = nn.Linear(self.hidden_dim, self.num_guidance_info)

        
    def forward(self, hidden_states):

        for block in self.attnblocks:
            hidden_states = block(hidden_states)
        #cls_token = hidden_states[:,0]
        #output = self.W(cls_token)
        output = self.W(hidden_states)

        output = torch.mean(output, dim=1)
        
        output = output*self.divide_out
        
        g_init = torch.sigmoid(output/self.init_g)*self.init_g +1
        
        return g_init
    

class AttentionModel2(nn.Module):
    
    def __init__(self,
                 init_g : float,
                 divide_out : float,
                 num_guidance_info : int,
                 num_layers : int,
                 hidden_dim : int,
                 heads : int=8,
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
        
        self.W = nn.Linear(self.hidden_dim, self.num_guidance_info)

        
    def forward(self, hidden_states):
        
        for block in self.attnblocks:
            hidden_states = block(hidden_states)
        cls_token = hidden_states[:,0]
        output = self.W(cls_token)
        output = output*self.divide_out
        
        out = self.relu(2*(torch.sigmoid(output/self.init_g)-0.5))
        out = out*self.init_g+1
        
        
        return out