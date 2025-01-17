from typing import Literal, Optional
import torch
import torch.nn as nn


class GuidanceModel(nn.Module):
    def __init__(
        self,
        init_g : float = 100.0,
        num_guidance_info : int = 3,
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
        
        self.init_G = torch.ones(num_guidance_info).to(device)
        self.init_G[0]*=init_g

        self.MLP_modules = []
        self.activate = activates[hidden_act]

        for _ in range(self.num_mlp_layers):
            self.MLP_modules.append(nn.Linear(self.in_size, self.in_size // 2))
            self.MLP_modules.append(self.activate)
            self.in_size = self.in_size // 2

        self.MLP = nn.Sequential(*self.MLP_modules)

        self.out = nn.Linear(self.in_size, self.num_guidance_info)

    def forward(self, x):

        out = self.MLP(x)
        out = self.out(out)
        out = torch.sigmoid(out)

        return out*self.init_G