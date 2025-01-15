from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging, BlipProcessor, BlipForConditionalGeneration
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import BaseOutput
from pnp_utils import *

# suppress partial model loading warning
logging.set_verbosity_error()


class PnPPipeline(nn.Module):
    def __init__(self,
                 n_timestep : int =50,
                 pnp_attn_t : float = 0.9,
                 pnp_f_t : float = 0.9,
                 sd_version : str ='2.1',
                 seed : int = 1,
                 latents_steps : int=1000,
                 device : str = 'cuda',
                 
                 ):
        super().__init__()

        self.device = device
        self.latents_steps = latents_steps
        self.seed = seed
        self.n_timestep=n_timestep
        self.pnp_attn_t = int(n_timestep*pnp_attn_t)
        self.pnp_f_t = int(n_timestep*pnp_f_t)

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')


        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        
        self.inversion_timesteps = reversed(self.scheduler.timesteps)
        self.scheduler.set_timesteps(self.n_timestep, device=self.device)
        
        
        
        self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.i2t_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(self.device)
        
        self.init_pnp(self.pnp_f_t, self.pnp_attn_t)
        
        
    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)
        
        
    def generate_prompt(self, image_dirs):
        
        images = [Image.open(img) for img in image_dirs]
        
        text = ["a photography of"]*len(images)
        inputs = self.image_processor(images, text, 
                                      return_tensors="pt").to(self.device, torch.float16)

        outputs = self.i2t_model.generate(**inputs)

        captions = [self.image_processor.decode(out, skip_special_tokens=True) for out in outputs]

        captions = [cap.replace(' at night','') for cap in captions]

        return captions
        
    def save_tensor(self, save_dir):
        scales_save = save_dir/f'scale-mean-std.pt'

        noise_scale = torch.tensor(self.noise_scale).unsqueeze(0)
        noise_mean = torch.tensor(self.noise_mean).unsqueeze(0)
        noise_std = torch.tensor(self.noise_std).unsqueeze(0)
        noise_info = torch.cat([noise_scale, noise_mean, noise_std])
        torch.save(noise_info, scales_save)


    @torch.no_grad()
    def get_text_embeds(self, prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img



    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = [self.load_source_latents_t(t,latent) for latent in self.source_latents_save_dirs]
        source_latents = torch.cat(source_latents).to(self.device)
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.negative_text_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        guide = self.guidance_scales[self.latents_steps-1-t]
        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)

        noise_pred = noise_pred_uncond + guide * (noise_pred_cond - noise_pred_uncond)

        # todo : noise scale check
        '''
        self.noise_scale.append(torch.mean(noise_pred**2).sqrt().item())
        self.noise_mean.append(torch.mean(noise_pred).item())
        self.noise_std.append(torch.std(noise_pred).item())'''
        
        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

        
    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
    
        return decoded_latent
    
    
    def load_source_latents_t(self, t, latents_path:Path):
        
        latents_t_path = latents_path/f'noisy_latents_{t}.pt'
        assert latents_t_path.exists(), f'Missing latents at t {t} path {latents_path.stem}'
        latents = torch.load(latents_t_path, weights_only=False)
        
        return latents
    
    
    def get_T_noise(self)-> torch.Tensor:
        
        init_timestep = self.scheduler.timesteps[0]
        
        #assert len(self.source_latents_save_dirs)==0, 'There is no source latents save directories'

        latents_paths = [latent/f'noisy_latents_{init_timestep}.pt' for latent in self.source_latents_save_dirs]
        noisy_latents = [torch.load(path, weights_only=False) for path in latents_paths]
        noisy_latents = torch.cat(noisy_latents).to(self.device, dtype=torch.float32)
        return noisy_latents
    
    def __call__(self,
                 image_dirs : Union[Path, List[Path]] = None,
                 conditions : Optional[Union[str, List[str]]] = None,
                 latents_save_root : Optional[str] = 'latents_forward',
                 guidance_scales : Optional[torch.Tensor] = None,
                 negative_prompt: Optional[str] = None,
                 ):
        # make list variable
        image_dirs = [image_dirs] if not isinstance(image_dirs, list) else image_dirs
        
        # define guidance scales
        if guidance_scales is None:
            alpha = self.scheduler.alphas_cumprod
            _guidance = 50*alpha
            self.guidance_scales = _guidance
        else:
            self.guidance_scales = guidance_scales
        
        # make latent root
        latent_save_root_dir = Path(latents_save_root)
        latent_save_root_dir.mkdir(exist_ok=True)
        
        # define batch size
        if image_dirs is not None and isinstance(image_dirs, Path):
            batch_size = 1
        elif image_dirs is not None and isinstance(image_dirs, list):
            batch_size = len(image_dirs)
        else:
            batch_size =1
            
        #generate prompt
        prompts = self.generate_prompt(image_dirs)
        
        #combine with condition
        if conditions is not None and isinstance(conditions, list):
            num_conditions = len(conditions)
            prompts = [prompts[i]+conditions[np.random.randint(0, num_conditions)] for i in range(batch_size)]
        elif conditions is not None and isinstance(conditions, str):
            prompts = [prompts[i]+conditions for i in range(batch_size)]
            
        #negative prompt
        negative_prompt = "" if negative_prompt==None else negative_prompt
            
        # text encoding
        self.text_embeds = self.get_text_embeds(prompts)
        negative_text_embeds : torch.Tensor = self.get_text_embeds(negative_prompt)
        pnp_guidance_embeds : torch.Tensor = self.get_text_embeds("")
        
        self.pnp_guidance_embeds = pnp_guidance_embeds.repeat(batch_size,1,1)
        self.negative_text_embeds = negative_text_embeds.repeat(batch_size,1,1)
        
        # to do: noise scale check
        '''self.noise_scale = []
        self.noise_mean = []
        self.noise_std = []'''
            
        # check latents, create latents
        last_step = self.inversion_timesteps[-1]
        
        self.source_latents_save_dirs = []
        
        for img_dir in image_dirs:
            source_latent_save_dir = latent_save_root_dir/img_dir.stem
            self.source_latents_save_dirs.append(source_latent_save_dir)
            
            source_latent_last_step_file = source_latent_save_dir/f'noisy_latents_{last_step}.pt'
            if source_latent_last_step_file.exists():
                print(f'There exist {img_dir.stem} latent files')
            else:
                source_latent_save_dir.mkdir(exist_ok=True)
                self.extract_latents(img_dir, source_latent_save_dir)
                
        # call latent zT
        zT = self.get_T_noise()
        
        # denoising
        decoded_latent = self.sample_loop(zT)
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
        image_pil = Image.open(image_path).convert("RGB").resize((512,512))
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

    images: Optional[List[Image.Image]]
    prompts: Optional[List[str]]