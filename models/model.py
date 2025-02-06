from typing import Literal, Optional,List
import torch
import torch.nn as nn
from models.attn_module import *


class GuidanceModel(nn.Module):
    def __init__(
        self,
        init_g : Optional[float] = None,
        num_guidance_info : int = 1,
        linear_in_size: Optional[int] = None,
        num_mlp_layers: int = 2,
        hidden_act: Literal["gelu", "mish", "selu"] = "gelu",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        activates = {"gelu": nn.GELU(), "mish": nn.Mish(), "silu": nn.SiLU()}
        
        self.device = device
        
        self.num_guidance_info = num_guidance_info
        self.in_size = linear_in_size
        self.hidden_act = hidden_act
        self.num_mlp_layers = num_mlp_layers
        
        self.init_G = 50.0 if init_g is None else init_g



        self.MLP_modules = []
        self.activate = activates[hidden_act]
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size))
            self.MLP_modules.append(self.activate)
        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)

        self.out = nn.Linear(self.in_size, self.num_guidance_info)

    def forward(self, x):

        out = self.MLP(x)
        out = self.out(out)
        out = torch.sigmoid(out/self.init_G)*self.init_G+1

        return out
    
    
class AttentionModel(nn.Module):
    
    def __init__(self,
                 hidden_dim : int,
                 heads : int,
                 init_g : float
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.init_g = init_g
        
        self.attn = Attention(self.hidden_dim, self.heads)
        self.ff = FeedForward(self.hidden_dim)
        
        self.W = nn.Linear(self.hidden_dim, 1)

        
    def forward(self, hidden_states):

        hidden_states = self.attn(hidden_states)
        hidden_states = self.ff(hidden_states)

        output = self.W(hidden_states)
        g_init = output[:,0]
        condition_portion = output[:,1:]
        
        g_init = torch.sigmoid(g_init/self.init_g)*self.init_g +1
        condition_portion = torch.softmax(condition_portion,dim=1)
        
        return g_init, condition_portion
    
    
class AttentionModelNoBLIP(nn.Module):
    
    def __init__(self,
                 hidden_dim : int,
                 heads : int,
                 init_g : float,
                 image_out : int = 2,
                 text_out : int =1
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.init_g = init_g
        self.image_out = image_out
        self.text_out = text_out
        
        self.attn = Attention(self.hidden_dim, self.heads)
        self.ff = FeedForward(self.hidden_dim)
        
        self.image_W = nn.Linear(self.hidden_dim, self.image_out)
        self.text_W = nn.Linear(self.hidden_dim, self.text_out)

        
    def forward(self, hidden_states):

        hidden_states = self.attn(hidden_states)
        hidden_states = self.ff(hidden_states)

        image_out = hidden_states[:,0]
        texts_out = hidden_states[:,1:]
        
        image_out = self.image_W(image_out)
        condition_portion = self.text_W(texts_out)
        
        g_init = image_out[:,0]
        origin_rate = image_out[:,1]
        
        g_init = torch.sigmoid(g_init/self.init_g)*self.init_g +1
        origin_rate = torch.sigmoid(origin_rate)
        condition_portion = torch.softmax(condition_portion,dim=1)
        
        return g_init, origin_rate, condition_portion
        