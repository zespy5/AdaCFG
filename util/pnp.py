from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import BaseOutput
from util.pnp_utils import *

# suppress partial model loading warning
logging.set_verbosity_error()


class PnPPipeline(nn.Module):
    def __init__(self,
                 n_timestep : int =50,
                 latents_steps : int=1000,
                 device : str = 'cuda',
                 pnp_attn_t : float = 0.9,
                 pnp_f_t : float = 0.9,
                 tensor_out : bool = False,
                 image_size : int = 512,
                 ):
        super().__init__()

        self.device = device
        self.latents_steps = latents_steps
        self.n_timestep=n_timestep
        self.pnp_attn_t = int(n_timestep*pnp_attn_t)
        self.pnp_f_t = int(n_timestep*pnp_f_t)
        self.tensor_out = tensor_out
        self.image_size = image_size
        

        model_key = "stabilityai/stable-diffusion-2-1-base"

        pipe = StableDiffusionPipeline.from_pretrained(model_key).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(self.latents_steps, device=self.device)
        self.inversion_timesteps = reversed(self.scheduler.timesteps)
        self.scheduler.set_timesteps(self.n_timestep, device=self.device)

        
        self.image_processor = None
        self.i2t_model = None
        
        self.init_pnp(self.pnp_f_t, self.pnp_attn_t)
        
        for p in self.vae.parameters():
            p.requires_grad=False
            
        for p in self.text_encoder.parameters():
            p.requires_grad=False
            
        for p in self.unet.parameters():
            p.requires_grad=False
            
        
    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)
       
        
        
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        return text_embeddings

    #@torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    
    def denoise_step(self, x, t, i):
        batch_size, channels, h, w = x.shape
        
        if self.image_latents is None:
            # register the time step and features in pnp injection modules
            source_latents = [self.load_source_latents_t(t,latent) for latent in self.source_latents_save_dirs]
            source_latents = torch.cat(source_latents).to(self.device)
        else:
            source_latents = self.image_latents[:,i]
            
        _xs = [x]*2
        latent_model_input = torch.cat([source_latents] + _xs)
        # latent_model_input [batch*(num_condition+2), 4, 64, 64]

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.negative_text_embeds, self.text_embeds], dim=0).to(self.device)
        # pnp_guidance_embeds [batch, 77, 1024]
        # negative_text_embeds [batch, 77, 1024]
        # text_embeds [batch*num_condition, 77, 1024]

        # apply the denoising network
        with torch.no_grad():
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']
        
        # perform guidance
        noise_pred = noise_pred.chunk(3)
        #origin_noise = noise_pred[0]
        
        noise_pred_uncond = noise_pred[1]
        noise_pred_conds = noise_pred[2]

        guide = self.guidance_scales[:,i].view(batch_size,1,1,1)
        noise_pred = noise_pred_uncond + guide*(noise_pred_conds - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        
        return denoised_latent

        
    def sample_loop(self, x):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            for i, t in enumerate(self.scheduler.timesteps):
                x = self.denoise_step(x, t, i)
            
            decoded_latent = self.decode_latent(x)
        return decoded_latent
    
    
    def load_source_latents_t(self, t, latents_path:Path):
        
        latents_t_path = latents_path/f'noisy_latents_{t}.pt'
        assert latents_t_path.exists(), f'Missing latents at t {t} path {latents_path.stem}'
        latents = torch.load(latents_t_path, weights_only=False, map_location=self.device)
        return latents
    
    
    def get_T_noise(self)-> torch.Tensor:
        
        init_timestep = self.scheduler.timesteps[0]
        
        latents_paths = [latent/f'noisy_latents_{init_timestep}.pt' for latent in self.source_latents_save_dirs]
        noisy_latents = [torch.load(path, weights_only=False, map_location=self.device) for path in latents_paths]
        noisy_latents = torch.cat(noisy_latents).to(self.device, dtype=torch.float32)
        return noisy_latents
    
    
    def __call__(self,
                 image_dirs : Union[Path, List[Path]] = None,
                 image_latents : Optional[torch.Tensor] = None,
                 
                 prompts : Optional[Union[str, List[str]]] = None,
                 prompts_embeddings : Optional[torch.Tensor] = None,
                 
                 negative_prompt: Optional[str] = None,
                 negative_prompt_embeddings : Optional[torch.Tensor] = None,
                 
                 guidance_scales : Optional[torch.Tensor] = None,
                 
                 latents_save_root : str = 'latents_forward',
                 ):

        register_condition_num(self)
        
        self.image_latents = image_latents
        self.prompts_embeddings = prompts_embeddings
        self.negative_prompt_embeddings = negative_prompt_embeddings

        if self.image_latents is None:
            
            assert image_dirs is not None, "image_dirs must not be None"
            # make list variable
            image_dirs = [image_dirs] if not isinstance(image_dirs, list) else image_dirs
            
            # make latent root
            latent_save_root_dir = Path(latents_save_root)
            latent_save_root_dir.mkdir(exist_ok=True)
    
            
            # check latents, create latents
            last_step = self.inversion_timesteps[-1]
            
            self.source_latents_save_dirs = []
            
            for img_dir in image_dirs:
                source_latent_save_dir = latent_save_root_dir/img_dir.stem
                self.source_latents_save_dirs.append(source_latent_save_dir)
                
                source_latent_last_step_file = source_latent_save_dir/f'noisy_latents_{last_step}.pt'
                if not source_latent_last_step_file.exists():
                    source_latent_save_dir.mkdir(exist_ok=True)
                    self.extract_latents(img_dir, source_latent_save_dir)
                    
            # call latent zT
            zT = self.get_T_noise()
            #zT = tensor[batch,4,64,64]
        else:
            zT = self.image_latents[:,0]
            
        batch_size = zT.shape[0]

        if self.prompts_embeddings is None:
            # text encoding
            self.text_embeds = self.get_text_embeds(prompts)
        else:
            self.text_embeds = self.prompts_embeddings
            
        
        if self.negative_prompt_embeddings is None:
            # text_embeds [batch_size*numcondition, 77, 768]
            #negative prompt
            negative_prompt = "" if negative_prompt==None else negative_prompt
            negative_text_embeds : torch.Tensor = self.get_text_embeds(negative_prompt)
            self.negative_text_embeds = negative_text_embeds.repeat(batch_size,1,1)
        else:
            self.negative_text_embeds = self.negative_prompt_embeddings
        
        pnp_guidance_embeds : torch.Tensor = self.get_text_embeds("")
        self.pnp_guidance_embeds = pnp_guidance_embeds.repeat(batch_size,1,1)
        
        # define guidance scales
        if guidance_scales is None:
            alpha = self.scheduler.alphas_cumprod.to(self.device)
            alpha = alpha[self.scheduler.timesteps]
            _guidance = 20*alpha.unsqueeze(0).repeat(batch_size,1).flip(1)
            self.guidance_scales  = _guidance
        else:
            self.guidance_scales = guidance_scales
        
        # denoising
        decoded_latent = self.sample_loop(zT)


        if self.tensor_out:
            return PnPPipelineOutput(images=decoded_latent, prompts=[""])
        
        edited_imgs = [T.ToPILImage()(latent) for latent in decoded_latent]
        
        return PnPPipelineOutput(images=edited_imgs, prompts=prompts)

        
    @torch.no_grad()
    def extract_latents(self, data_path, save_path):
        cond = self.get_text_embeds("")
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)
        register_time(self, 0)
        self.ddim_inversion(cond, latent, save_path) 
        
    
    def load_img(self, image_path):
        image_pil = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents
        
    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            for i, t in enumerate(tqdm(self.inversion_timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[self.inversion_timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_save_path = save_path/f'noisy_latents_{t}.pt'
                if save_latents:
                    torch.save(latent, latent_save_path)
    
    
@dataclass
class PnPPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        prompts (`List[str]`)
    """

    images: Optional[Union[List[Image.Image], torch.Tensor]]
    prompts: Optional[List[str]]