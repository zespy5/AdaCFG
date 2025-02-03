import torch
from diffusers import DDIMScheduler


class GuidanceScheduler(DDIMScheduler):
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 n_timestep: int = 50,
                 device : str = 'cuda',
                 num_condition: int = 3,
                 ):
        super().__init__(num_train_timesteps=num_train_timesteps)
        
        self.num_train_timesteps = num_train_timesteps
        self.set_timesteps(n_timestep)
        self.device = device
        self.num_condition= num_condition
        

    @torch.no_grad()
    def schedule(self, beta_start:float = 0.00085, beta_end:float = 0.012):
        
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32)**2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return alphas_cumprod
        
        
    def get_guidance_scales(self, schedule_info : torch.Tensor)->torch.Tensor:
        
        batch_size, _ = schedule_info.shape

        timesteps = self.timesteps.repeat(batch_size, self.num_condition,1)
        
        scheduler = self.schedule().unsqueeze(0).repeat(batch_size, self.num_condition,1)

        selected_schedulers = torch.gather(scheduler,2, timesteps)
        selected_schedulers = selected_schedulers.flip(2).to(self.device)
        
        schedulers = schedule_info.unsqueeze(-1)*selected_schedulers
        #schedulers = torch.where(schedulers<1, 1.0, schedulers)

        return schedulers
