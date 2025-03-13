from typing import Any, Callable, Dict, List, Optional, Union,Literal
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import torch.nn.functional as F
from util.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode,ToPILImage
BICUBIC = InterpolationMode.BICUBIC
BILINEAR = InterpolationMode.BILINEAR

class Loss(nn.Module):
    
    def __init__(self,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 0.5,
                 lambda_structure: float = 2.5,
                 dino_threshold : float = 0.2,
                 num_condition : int = 3,
                 pnp_injection_rate : float = 0.9,
                 device :str = 'cuda',
                 clip_ds_use : bool = True,
                 negative_clip_use : bool = True,
                 gradient : Literal['increase', 'decrease', 'constant'] = 'increase',
                 schedule_method : Literal['cosine', 'linear'] = 'cosine',
                 **kwargs
                 ):
        super().__init__()
        
        self.device = device
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_structure = lambda_structure
        self.dino_threshold = dino_threshold
        self.num_condition = num_condition
        self.pnp_injection_rate = pnp_injection_rate
        self.clip_ds_use = clip_ds_use
        self.negative_clip_use = negative_clip_use
        
        
        self.pipeline = PnPPipeline(device=self.device,
                                    pnp_attn_t=self.pnp_injection_rate,
                                    pnp_f_t=self.pnp_injection_rate,
                                    tensor_out=True)
        
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
        
    def dino_transform(self, n_px= 256,interpolation=BICUBIC, max_size=None):
        return Compose([
            Resize(n_px, interpolation=interpolation, max_size=max_size),
            CenterCrop(224),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
    def clip_loss(self, gen_images, prompts):
        batch = gen_images.shape[0]
        prompt_batch = prompts.shape[0]
          
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        if batch != prompt_batch:
            assert prompt_batch%batch == 0 and prompt_batch//batch==self.num_condition, 'miss match from_prompts and to_prompts size'
            img_features = img_features.repeat(self.num_condition,1)
            clip_cs = F.cosine_similarity(prompts, img_features, dim=1).view(batch,self.num_condition)
            loss = (1-clip_cs).sum(dim=1)
        else :
            clip_cs = F.cosine_similarity(prompts, img_features, dim=1)
            loss = (1-clip_cs)
        
        loss = loss.view(-1).mean()
        
        return loss, clip_cs.view(-1)
    
    #todo matching the batch size
    def clip_ds_loss(self, real_images,  gen_images, from_prompts, to_prompts):
        to_batch = to_prompts.shape[0]
        from_batch = from_prompts.shape[0]
        
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))
        
        
        if to_batch != from_batch:
            assert to_batch%from_batch == 0 and to_batch//from_batch==self.num_condition, 'miss match from_prompts and to_prompts size'
            
            real_images = real_images.repeat(self.num_condition, 1)
            img_features = img_features.repeat(self.num_condition,1)
            from_prompts = from_prompts.repeat(self.num_condition,1)
        
        
        delta_text_features = to_prompts-from_prompts
        delta_image_features = img_features-real_images

        clip_cs = F.cosine_similarity(delta_text_features, delta_image_features, dim=1)
        loss = (1-clip_cs)
        loss = loss.view(-1).mean()
        return loss, clip_cs
    
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
                clip_real_image_embedding,
                from_clip_embedding,
                to_clip_embedding,
                model_input_embedding,
                image_latents,
                sd_prompt_embedding,
                origin_alpha:Optional[Union[torch.Tensor,float]]=None,
                g_init:Optional[torch.Tensor]=None,
                g_portion:Optional[torch.Tensor]= None,
                ):
        
        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(g_init)

        outputs = self.pipeline(image_latents=image_latents,
                                prompts_embeddings=sd_prompt_embedding,
                                negative_prompt=self.negative_prompt, #todo: negative_prompt embedding
                                num_condition=self.num_condition,
                                origin_alpha=origin_alpha,
                                guidance_scales=scheduled_guidance,
                                guidance_portion=g_portion,
                                )
        
        gen_images = outputs.images
        batch_size = gen_images.shape[0]

        

        if self.clip_ds_use:
            clip_loss, clip_cs = self.clip_ds_loss(clip_real_image_embedding, gen_images, from_clip_embedding, to_clip_embedding)
        else:
            clip_loss, clip_cs = self.clip_loss(gen_images, to_clip_embedding)

        
        if self.negative_clip_use:
            negative_clip_embedding = self.prompt_embeds(self.negative_prompt)
            negative_clip_embedding = negative_clip_embedding.repeat(batch_size,1)
            negative_clip_loss, _ = self.clip_loss(gen_images, negative_clip_embedding)
            negative_clip_loss = 1-negative_clip_loss
            clip_loss = clip_loss+negative_clip_loss
            
            
        structure_loss, dino_cs = self.structure_loss_func(real_image_tensor, gen_images)
        
        if self.dino_threshold > 0:
            structure_loss = torch.sqrt((structure_loss - self.dino_threshold)**2)
        
        
        loss = self.lambda_text*clip_loss + self.lambda_structure*structure_loss

        return loss, gen_images, clip_cs, dino_cs
            
            
            

class BLIPLoss(nn.Module):
    
    def __init__(self,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 clip_negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint, black and white',
                 lambda_text : float = 1.0,
                 lambda_blip : float = 0.1,
                 lambda_structure: float = 0.1,
                 lambda_negative : float = 1.0,
                 dino_threshold : float = 0.2,
                 num_condition : int = 3,
                 pnp_injection_rate : float = 0.9,
                 device :str = 'cuda',
                 dino_loss_use : bool = True,
                 clip_ds_use : bool = True,
                 negative_clip_use : bool = True,
                 gradient : Literal['increase', 'decrease', 'constant'] = 'increase',
                 schedule_method : Literal['cosine', 'linear'] = 'cosine',
                 **kwargs
                 ):
        super().__init__()
        
        self.device = device
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_blip = lambda_blip
        self.lambda_structure = lambda_structure
        self.lambda_negative = lambda_negative
        self.dino_threshold = dino_threshold
        self.num_condition = num_condition
        self.pnp_injection_rate = pnp_injection_rate
        self.dino_loss_use = dino_loss_use
        self.clip_ds_use = clip_ds_use
        self.negative_clip_use = negative_clip_use
        self.clip_negative_prompt = clip_negative_prompt
        
        
        self.pipeline = PnPPipeline(device=self.device,
                                    pnp_attn_t=self.pnp_injection_rate,
                                    pnp_f_t=self.pnp_injection_rate,
                                    tensor_out=True)
        
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
        
        self.negative_clip_embedding = self.prompt_embeds(self.clip_negative_prompt)
   
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
        
    def dino_transform(self, n_px= 256,interpolation=BICUBIC, max_size=None):
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
                from_clip_embedding,
                to_clip_embedding,
                image_latents,
                sd_prompt_embedding,
                origin_alpha:Optional[Union[torch.Tensor,float]]=None,
                g_init:Optional[torch.Tensor]=None,
                g_portion:Optional[torch.Tensor]= None,
                ):
        
        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(g_init)

        outputs = self.pipeline(image_latents=image_latents,
                                prompts_embeddings=sd_prompt_embedding,
                                negative_prompt=self.negative_prompt, #todo: negative_prompt embedding
                                num_condition=self.num_condition,
                                origin_alpha=origin_alpha,
                                guidance_scales=scheduled_guidance,
                                guidance_portion=g_portion,
                                )
        
        gen_images = outputs.images
        batch_size = gen_images.shape[0]

        

        from_clip_loss, from_clip_cs = self.clip_loss(gen_images, from_clip_embedding)
        to_clip_loss, clip_cs = self.clip_loss(gen_images, to_clip_embedding)
        
        clip_loss = self.lambda_blip*from_clip_loss + self.lambda_text*to_clip_loss
        if self.negative_clip_use:
            negative_clip_embedding = self.negative_clip_embedding.repeat(batch_size,1)
            negative_clip_loss, _ = self.clip_loss(gen_images, negative_clip_embedding)
            negative_clip_loss = 1-negative_clip_loss
            clip_loss = clip_loss+negative_clip_loss*self.lambda_negative
            
            
        structure_loss, dino_cs = self.structure_loss_func(real_image_tensor, gen_images)
        
        if self.dino_threshold > 0:
            structure_loss = torch.sqrt((structure_loss - self.dino_threshold)**2)
        
        
        loss = clip_loss + self.lambda_structure*structure_loss

        clip_cs = torch.cat([from_clip_cs, clip_cs])
        return loss, gen_images, clip_cs, dino_cs
            
            