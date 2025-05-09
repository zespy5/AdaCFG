import torch
from diffusers import DDIMScheduler
import math
from typing import Literal
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class GuidanceScheduler(DDIMScheduler):
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 n_timestep: int = 50,
                 lower_bound: float = 1.0,
                 gradient : Literal['increase', 'decrease', 'constant', 'sine'] = 'increase',
                 schedule_method : Literal['cosine', 'linear'] = 'cosine',
                 device : str = 'cuda',
                 ):
        super().__init__(num_train_timesteps=num_train_timesteps)
        
        self.num_train_timesteps = num_train_timesteps
        self.set_timesteps(n_timestep)
        self.device = device
        self.gradient = gradient
        self.schedule_method = schedule_method
        self.lower_bound = lower_bound
        

    @torch.no_grad()
    def schedule(self,beta_start:float = 0.00085, beta_end:float = 0.012):
        
        linear = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32)**2
        cosine = betas_for_alpha_bar(self.num_train_timesteps)
        betas = cosine if self.schedule_method=='cosine' else linear
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return alphas_cumprod
    
    @torch.no_grad()
    def sine_schedule(self):
        
        x = torch.linspace(0, torch.pi, self.num_train_timesteps, dtype=torch.float32)
        alphas = torch.sin(x)**2
        
        return alphas
        
        
        
    def get_guidance_scales(self, schedule_info : torch.Tensor)->torch.Tensor:
        
        if self.gradient=='constant':
            return schedule_info.repeat(1,self.num_inference_steps)
        
        batch_size, _ = schedule_info.shape

        timesteps = self.timesteps.repeat(batch_size,1)
        
        if self.gradient=='sine':
            scheduler = self.sine_schedule().unsqueeze(0).repeat(batch_size,1)
        else:
            scheduler = self.schedule().unsqueeze(0).repeat(batch_size,1)

        selected_schedulers = torch.gather(scheduler,1, timesteps).to(self.device)
        selected_schedulers = selected_schedulers.flip(1) if self.gradient=='decrease' else selected_schedulers

        schedulers = schedule_info*selected_schedulers
        schedulers = torch.where(schedulers<self.lower_bound, self.lower_bound, schedulers)

        return schedulers
