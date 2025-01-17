import torch
from diffusers import DDIMScheduler


class GuidanceScheduler(DDIMScheduler):
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 n_timestep: int = 50,
                 ):
        super().__init__(num_train_timesteps=num_train_timesteps)
        
        self.num_train_timesteps = num_train_timesteps
        self.set_timesteps(n_timestep)
        

        
    def schedule(self, init_num, beta_start, beta_end):
        
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return alphas_cumprod*init_num
        
        
    def get_guidance_scales(self, schedule_info : torch.Tensor)->torch.Tensor:
        
        batch_size, info_len = schedule_info.shape
        assert info_len==3, f"the schedule information length is 3, but this information length is {info_len}"
        timesteps = self.timesteps.repeat(batch_size,1)
        schedule_infos = schedule_info.cpu().numpy()
        
        schedulers = [self.schedule(s[0], s[1], s[2]).unsqueeze(0) for s in schedule_infos]
        schedulers = torch.cat(schedulers)

        selected_schedulers = torch.gather(schedulers,1, timesteps)
        selected_schedulers = selected_schedulers.flip(1)

        return selected_schedulers
    
if __name__=="__main__":
    
    tensor = torch.tensor([[50,0.0001,0.012],
                       [50,0.0001,0.013]])
    scheduler = GuidanceScheduler(batch_size=2)

    scheduler.get_guidance_scales(tensor)