from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import torch.nn.functional as F
from utils.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from pathlib import Path

class Loss(nn.Module):
    
    def __init__(self,
                 conditions : Optional[Union[str, List[str]]]= None,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 0.5,
                 lambda_structure: float = 2.5,
                 lambda_reg : float = 0.1,
                 device :str = 'cuda',
                 data_root : str='image_data'):
        super().__init__()
        
        self.device = device
        self.conditions = conditions
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_structure = lambda_structure
        self.lambda_reg = lambda_reg
        self.data_root = Path(data_root)
        
        self.pipeline = PnPPipeline(generate_condition_prompt=False)
        self.guidance_scheduler = GuidanceScheduler()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

    @torch.no_grad()
    def prompt_embeds(self, prompts):
        clip_inputs = self.clip_processor(text=prompts,
                                            return_tensors="pt", 
                                            padding=True)
        text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                            clip_inputs["attention_mask"].to(self.device))
            
        return text_features
        
    
    
    @torch.no_grad()
    def generate_edited_image(self, image_dirs, prompts, guidance_info):

        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(guidance_info)
    
        outputs = self.pipeline(image_dirs=image_dirs,
                                prompts = prompts,
                                guidance_scales=scheduled_guidance,
                                negative_prompt=self.negative_prompt)
        gen_images = outputs.images
        edited_prompt = outputs.prompts
        
        return gen_images, edited_prompt
            
    def clip_loss(self, gen_images, prompts):
        
        with torch.no_grad():
            clip_inputs = self.clip_processor(text=prompts, images=gen_images, 
                                          return_tensors="pt", padding=True)
            img_features = self.clip_model.get_image_features(clip_inputs["pixel_values"].to(self.device))
            text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                              clip_inputs["attention_mask"].to(self.device))
            
            
            
        loss = F.mse_loss(img_features, text_features)
        return loss
    
    def dino_loss(self, real_images, gen_images):
        
        with torch.no_grad():
            images = real_images + gen_images
            inputs = self.dino_processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.dino_model(**inputs).last_hidden_state
            
            reals, gens = outputs.chunk(2)
        
        loss = F.mse_loss(reals, gens)
        
        return loss
    
    def forward(self, 
                image_idxs: torch.Tensor, 
                prompts,
                guidance_info:torch.Tensor):
        
        image_dirs = [self.data_root/f'{idx:03}.png' for idx in image_idxs.numpy()]
        real_images = [Image.open(img).convert('RGB') for img in image_dirs]
        
        gen_images, _ = self.generate_edited_image(image_dirs=image_dirs, 
                                                   prompts=prompts,
                                                   guidance_info=guidance_info)
        text_loss = self.clip_loss(gen_images, prompts)
        structure_loss = self.dino_loss(real_images, gen_images)
        
        loss = self.lambda_text*text_loss + self.lambda_structure*structure_loss
        reg_loss = 0.5*guidance_info.pow(2).sum()
        
        loss += self.lambda_reg*reg_loss
        
        return loss, gen_images
            
            