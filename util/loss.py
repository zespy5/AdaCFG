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
from torchvision.transforms import InterpolationMode
from metrics.extractor import VitExtractor
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
                 guidance_schedule_use :bool = True,
                 device :str = 'cuda',
                 dino_loss_use : bool = True,
                 clip_ds_use : bool = True,
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
        self.guidance_schedule_use = guidance_schedule_use
        self.dino_loss_use = dino_loss_use
        self.clip_ds_use = clip_ds_use
        
        
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
        
        if self.dino_loss_use:
            self.structure_transform = self.dino_transform()
            self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
            self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            for p in self.dino_model.parameters():
                p.requires_grad=False 
        else :
            self.structure_transform = self.dino_transform(224,BILINEAR,480)
            self.vit_extractor = VitExtractor('dino_vitb16', self.device)
            for p in self.vit_extractor.model.parameters():
                p.requires_grad=False
        
        self.structure_loss_func = self.dino_loss if self.dino_loss_use else self.keys_self_sim_score
   
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
        return loss, clip_cs
    
    def clip_ds_loss(self, real_images,  gen_images, from_prompts, to_prompts):
        prompts = from_prompts + to_prompts
        with torch.no_grad():
            clip_inputs = self.clip_processor(text=prompts,images=real_images, return_tensors="pt", padding=True)
            real_image_features = self.clip_model.get_image_features(clip_inputs['pixel_values'].to(self.device))
            text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                              clip_inputs["attention_mask"].to(self.device))
            from_text_feaures, to_text_features = text_features.chunk(2)
            delta_text_features = to_text_features-from_text_feaures


        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))
        
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
    
    def keys_self_sim_score(self, real, gen):
        batch_size = len(gen)
        
        with torch.no_grad():
            real_inputs = self.structure_transform(real).to(self.device)
            real_outputs = self.vit_extractor.get_keys_self_sim_from_input(real_inputs, 11)
            real_t = real_outputs.view(batch_size,-1)
            
        gen_inputs = self.structure_transform(gen).to(self.device)
        gen_outputs = self.vit_extractor.get_keys_self_sim_from_input(gen_inputs, 11)
        gen_t = gen_outputs.view(batch_size,-1)
           
        ksss_cs = F.cosine_similarity(real_t, gen_t, dim=1)
        loss = (1-ksss_cs)
        loss = loss.view(-1).mean()
        
        return loss, ksss_cs
    
    def forward(self,
                real_image_tensor,
                clip_real_image_embedding,
                from_clip_embedding,
                to_clip_embedding,
                model_input_embedding,
                image_latents,
                prompt_embeddings,
                real_images, 
                origin_alpha:Optional[Union[torch.Tensor,float]]=None,
                g_init:Optional[torch.Tensor]=None,
                g_portion:Optional[torch.Tensor]= None,
                ):
        
        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(g_init)
        
        outputs = self.pipeline(image_latents=image_latents,
                                prompts_embeddings=prompt_embeddings,
                                negative_prompt=self.negative_prompt,#todo: negative_prompt embedding
                                num_condition=self.num_condition,
                                origin_alpha=origin_alpha,
                                guidance_scales=scheduled_guidance,
                                guidance_portion=g_portion,
                                )
        
        gen_images = outputs.images

        clip_cses = []
        
        if self.clip_ds_use:
            clip_loss, clip_cs = self.clip_ds_loss(clip_real_image_embedding, gen_images, from_clip_embedding, to_clip_embedding)
            clip_cses.append(clip_cs.squeeze())
        else:
            clip_loss, clip_cs = self.clip_loss(gen_images, model_input_embedding)
            clip_cses.append(clip_cs.squeeze())


        structure_loss, dino_cs = self.structure_loss_func(real_image_tensor, gen_images)

        #threshold = torch.ones_like(structure_loss)*self.dino_threshold
        #thresholded_structure_loss = torch.max(threshold,structure_loss)
        thresholded_structure_loss = torch.sqrt((structure_loss - self.dino_threshold)**2)

        loss = self.lambda_text*clip_loss + self.lambda_structure*thresholded_structure_loss
        # todo  : 0.5*g_portion**2
        
        return loss, gen_images, clip_cses, dino_cs.squeeze()
            
            