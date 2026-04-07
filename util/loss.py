from typing import Optional, Literal
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import torch.nn.functional as F
from util.guidance_scheduler import GuidanceScheduler
from util.pnp import PnPPipeline
from util.ip2p import InstructPix2PixPipeline
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode,ToPILImage
BICUBIC = InterpolationMode.BICUBIC
BILINEAR = InterpolationMode.BILINEAR

            
        
            
    
class IP2PLoss(nn.Module):
    
    def __init__(self,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 1.0,
                 lambda_structure: float = 0.1,
                 lambda_mean : float = 0.1,
                 lambda_negative : float = 1.0,
                 n_timestep : int = 50,
                 image_guidance : float = 2.0,
                 devide_guide : float = 5.0,
                 negative_clip_use : bool = True,
                 gradient : Literal['increase', 'decrease', 'constant'] = 'increase',
                 schedule_method : Literal['cosine', 'linear'] = 'cosine',
                 device :str = 'cuda',
                 **kwargs
                 ):
        super().__init__()
        
        self.device = device
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_mean = lambda_mean
        self.lambda_structure = lambda_structure
        self.lambda_negative = lambda_negative
        self.negative_clip_use = negative_clip_use
        self.n_timestep = n_timestep
        self.image_guidance = image_guidance
        self.devide_guide = devide_guide
        model_id = "timbrooks/instruct-pix2pix"
        self.pipeline = InstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None).to(device)
        
        for p in self.pipeline.vae.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False
            
        self.guidance_scheduler = GuidanceScheduler(gradient=gradient,
                                                    schedule_method=schedule_method,
                                                    n_timestep=self.n_timestep,
                                                    device=self.device)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        for p in self.clip_model.parameters():
            p.requires_grad=False
        
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        for p in self.dino_model.parameters():
            p.requires_grad=False 
        
        self.structure_transform = self.dino_transform()
        self.structure_loss_func = self.dino_loss
        
        self.negative_clip_embedding = self.prompt_embeds(self.negative_prompt)
   
    @torch.no_grad()
    def prompt_embeds(self, prompts):
        clip_inputs = self.clip_processor(text=prompts,
                                          return_tensors="pt", 
                                          padding=True)
        text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                            clip_inputs["attention_mask"].to(self.device))
            
        return text_features
    
    @torch.no_grad()
    def image_clip_embeds(self, image):
        clip_inputs = self.clip_processor(images=image,
                                          return_tensors="pt", 
                                          padding=True)
        image_features = self.clip_model.get_image_features(clip_inputs["pixel_values"].to(self.device))
            
        return image_features
        
    
    def clip_transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def dino_transform(self, n_px= 224,interpolation=BICUBIC, max_size=None):
        return Compose([
            Resize(n_px, interpolation=interpolation, max_size=max_size),
            CenterCrop(224),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
    def clip_loss(self, gen_images, prompts):   
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        clip_cs = F.cosine_similarity(prompts, img_features, dim=1)
        loss = (1-clip_cs)
        
        loss = loss.view(-1).mean()
        
        return loss, clip_cs.view(-1)
    
    def neg_clip_loss(self, gen_images, prompts):   
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        clip_cs = F.cosine_similarity(prompts, img_features, dim=1)
        loss = clip_cs.view(-1).mean()
        
        return loss, clip_cs.view(-1)
    
    def dino_loss(self, real_images, gen_images):
        batch = gen_images.shape[0]
        with torch.no_grad():
            real_inputs = self.structure_transform(real_images).to(self.device)
            real_outputs = self.dino_model(real_inputs).last_hidden_state
            real_outputs = real_outputs.view(batch, -1)
        
        gen_inputs = self.structure_transform(gen_images).to(self.device)
        gen_outputs = self.dino_model(gen_inputs).last_hidden_state
        gen_outputs = gen_outputs.view(batch, -1)
        
        dino_cs = F.cosine_similarity(real_outputs, gen_outputs, dim=1)
        loss = (1-dino_cs)
        loss = loss.view(-1).mean()
        
        return loss, dino_cs
    
    
    def forward(self,
                real_image_tensor,
                condition_mean,
                sd_prompt_embedding,
                to_clip_embedding,
                g_init:Optional[torch.Tensor]=None,
                velocity:Optional[torch.Tensor]= None,
                ):
        
        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(g_init, velocity)

        outputs = self.pipeline(prompt_embeds=sd_prompt_embedding,
                                image=real_image_tensor,
                                num_inference_steps=self.n_timestep,
                                guidance_scale=scheduled_guidance,
                                image_guidance_scale=self.image_guidance,
                                devide_guide = self.devide_guide,
                                output_type='pt')
        
        gen_images = outputs.images
        batch_size = gen_images.shape[0]

        to_clip_loss, clip_cs = self.clip_loss(gen_images, to_clip_embedding)
        clip_loss = self.lambda_text*to_clip_loss
        
        if self.lambda_mean > 0:
            mean_clip_loss, mean_clip_cs = self.clip_loss(gen_images, condition_mean)
            clip_loss = clip_loss+self.lambda_mean*mean_clip_loss
        
        if self.negative_clip_use:
            negative_clip_embedding = self.negative_clip_embedding.repeat(batch_size,1)
            negative_clip_loss, _ = self.neg_clip_loss(gen_images, negative_clip_embedding)
            clip_loss = clip_loss+negative_clip_loss*self.lambda_negative 
            
            
        structure_loss, dino_cs = self.structure_loss_func(real_image_tensor, gen_images)
        
        loss = clip_loss + self.lambda_structure*structure_loss

        clip_cs = torch.cat([mean_clip_cs, clip_cs])
        return loss, gen_images, clip_cs, dino_cs
    
    
class VVLoss(nn.Module):
    
    def __init__(self,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 1.0,
                 lambda_structure: float = 0.1,
                 lambda_mean : float = 0.1,
                 lambda_negative : float = 1.0,
                 pnp_injection_rate : float = 0.9,
                 pnp_res_injection_rate : float = 0.8,
                 negative_clip_use : bool = True,
                 image_size : int = 512,
                 gradient : Literal['increase', 'decrease', 'constant'] = 'increase',
                 schedule_method : Literal['cosine', 'linear'] = 'cosine',
                 n_timestep : int = 50,
                 latents_steps : int = 50,
                 device :str = 'cuda',
                 **kwargs
                 ):
        super().__init__()
        
        self.device = device
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_mean = lambda_mean
        self.lambda_structure = lambda_structure
        self.lambda_negative = lambda_negative
        self.pnp_injection_rate = pnp_injection_rate
        self.pnp_res_injection_rate = pnp_res_injection_rate
        self.negative_clip_use = negative_clip_use
        
        self.n_timestep = n_timestep
        self.latents_steps = latents_steps
        
        
        self.pipeline = PnPPipeline(n_timestep=self.n_timestep,
                                    latents_steps=self.latents_steps,
                                    device=self.device,
                                    pnp_attn_t=self.pnp_injection_rate,
                                    pnp_f_t=self.pnp_res_injection_rate,
                                    tensor_out=True,
                                    image_size=image_size)
        
        self.guidance_scheduler = GuidanceScheduler(gradient=gradient,
                                                    schedule_method=schedule_method,
                                                    device=self.device)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        for p in self.clip_model.parameters():
            p.requires_grad=False
        
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        for p in self.dino_model.parameters():
            p.requires_grad=False 
        
        self.structure_transform = self.dino_transform()
        self.structure_loss_func = self.dino_loss
        
        self.negative_clip_embedding = self.prompt_embeds(self.negative_prompt)
   
    @torch.no_grad()
    def prompt_embeds(self, prompts):
        clip_inputs = self.clip_processor(text=prompts,
                                          return_tensors="pt", 
                                          padding=True)
        text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                            clip_inputs["attention_mask"].to(self.device))
            
        return text_features
    
    @torch.no_grad()
    def image_clip_embeds(self, image):
        clip_inputs = self.clip_processor(images=image,
                                          return_tensors="pt", 
                                          padding=True)
        image_features = self.clip_model.get_image_features(clip_inputs["pixel_values"].to(self.device))
            
        return image_features
        
    
    def clip_transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def dino_transform(self, n_px= 224,interpolation=BICUBIC, max_size=None):
        return Compose([
            Resize(n_px, interpolation=interpolation, max_size=max_size),
            CenterCrop(224),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
    def clip_loss(self, gen_images, prompts):   
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        clip_cs = F.cosine_similarity(prompts, img_features, dim=1)
        loss = (1-clip_cs)
        
        loss = loss.view(-1).mean()
        
        return loss, clip_cs.view(-1)
    
    def neg_clip_loss(self, gen_images, prompts):   
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        clip_cs = F.cosine_similarity(prompts, img_features, dim=1)
        loss = clip_cs.view(-1).mean()
        
        return loss, clip_cs.view(-1)
    
    
    def dino_loss(self, real_images, gen_images):
        batch = gen_images.shape[0]
        with torch.no_grad():
            real_inputs = self.structure_transform(real_images).to(self.device)
            real_outputs = self.dino_model(real_inputs).last_hidden_state
            real_outputs = real_outputs.view(batch, -1)
        
        gen_inputs = self.structure_transform(gen_images).to(self.device)
        gen_outputs = self.dino_model(gen_inputs).last_hidden_state
        gen_outputs = gen_outputs.view(batch, -1)
        
        dino_cs = F.cosine_similarity(real_outputs, gen_outputs, dim=1)
        loss = (1-dino_cs)
        loss = loss.view(-1).mean()
        
        return loss, dino_cs
    
    
    def forward(self,
                real_image_tensor,
                condition_mean,
                image_latents,
                sd_prompt_embedding,
                to_clip_embedding,
                g_init:Optional[torch.Tensor]=None,
                velocity:Optional[torch.Tensor]= None,
                ):
        
        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(g_init, velocity)

        outputs = self.pipeline(image_latents=image_latents,
                                prompts_embeddings=sd_prompt_embedding,
                                negative_prompt=self.negative_prompt, #todo: negative_prompt embedding
                                guidance_scales=scheduled_guidance)
        
        gen_images = outputs.images
        batch_size = gen_images.shape[0]
        
        to_clip_loss, clip_cs = self.clip_loss(gen_images, to_clip_embedding)
        clip_loss = self.lambda_text*to_clip_loss
        
        if self.lambda_mean > 0:
            mean_clip_loss, mean_clip_cs = self.clip_loss(gen_images, condition_mean)
            clip_loss = clip_loss+self.lambda_mean*mean_clip_loss
        
        if self.negative_clip_use:
            negative_clip_embedding = self.negative_clip_embedding.repeat(batch_size,1)
            negative_clip_loss, _ = self.neg_clip_loss(gen_images, negative_clip_embedding)
            clip_loss = clip_loss+negative_clip_loss*self.lambda_negative 
            
        structure_loss, dino_cs = self.structure_loss_func(real_image_tensor, gen_images)
        
        loss = clip_loss + self.lambda_structure*structure_loss

        #clip_cs = torch.cat([mean_clip_cs, clip_cs])
        return loss, gen_images, clip_cs, dino_cs
    