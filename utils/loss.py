from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import torch.nn.functional as F
from utils.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

class Loss(nn.Module):
    
    def __init__(self,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 0.5,
                 lambda_structure: float = 2.5,
                 device :str = 'cuda',
                 data_root : str='image_data/train',
                 save_image_path : Optional[str]= None,
                 latents_save_root : str = 'latents_forward',
                 dino_threshold : float = 0.2,
                 num_condition : int = 3,
                 generate_condition_prompt: bool =False,
                 ):
        super().__init__()
        
        self.device = device
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_structure = lambda_structure
        self.save_image_path = save_image_path
        self.latents_save_root = latents_save_root
        self.dino_threshold = dino_threshold
        self.num_condition = num_condition
        self.generate_condition_prompt = generate_condition_prompt
        
        self.data_root = Path(data_root)
        
        self.pipeline = PnPPipeline(generate_condition_prompt=self.generate_condition_prompt,
                                    device=self.device,
                                    tensor_out=True,
                                    train_mode=True)
        self.guidance_scheduler = GuidanceScheduler(device=self.device,
                                                    num_condition=self.num_condition)
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

        for p in self.clip_model.parameters():
            p.requires_grad=False
        
        for p in self.dino_model.parameters():
            p.requires_grad=False
            
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
        
    def generate_edited_image(self, image_dirs, prompts, guidance_info):

        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(guidance_info)

        outputs = self.pipeline(num_condition=self.num_condition,
                                image_dirs=image_dirs,
                                prompts = prompts,
                                guidance_scales=scheduled_guidance,
                                negative_prompt=self.negative_prompt,
                                latents_save_root=self.latents_save_root)
        gen_images = outputs.images
        edited_prompt = outputs.prompts
        
        return gen_images, edited_prompt
    
    def clip_transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def dino_transform(self):
        return Compose([
            Resize(256, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
    def clip_loss(self, gen_images, prompts):
        
        with torch.no_grad():
            clip_inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                              clip_inputs["attention_mask"].to(self.device))
        
        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        #loss = F.mse_loss(img_features, text_features)
        loss = (1-F.cosine_similarity(text_features, img_features, dim=1))
        loss = loss.view(-1).mean()
        return loss
    
    def dino_loss(self, real_images, gen_images):
        batch = gen_images.shape[0]
        with torch.no_grad():
            real_inputs = self.dino_processor(images=real_images, return_tensors="pt").to(self.device)
            real_outputs = self.dino_model(**real_inputs).last_hidden_state

        gen_inputs = self.dino_transform()(gen_images).to(self.device)
        gen_outputs = self.dino_model(gen_inputs).last_hidden_state
        
        real_outputs = real_outputs.view(batch, -1)
        gen_outputs = gen_outputs.view(batch, -1)
        
        #loss = F.mse_loss(real_outputs, gen_outputs)
        loss = (1-F.cosine_similarity(real_outputs, gen_outputs, dim=1))
        loss = loss.view(-1).mean()
        return loss
    
    def forward(self, 
                image_dirs,
                real_images, 
                prompts,
                guidance_info:torch.Tensor):

        gen_images, prompt_c = self.generate_edited_image(image_dirs=image_dirs, 
                                                   prompts=prompts,
                                                   guidance_info=guidance_info)

        text_losses = [self.clip_loss(gen_images, p).unsqueeze(0) for p in prompts]

        text_loss = torch.cat(text_losses).sum()

        structure_loss = self.dino_loss(real_images, gen_images)

        threshold = torch.ones_like(structure_loss)*self.dino_threshold
        thresholded_structure_loss = self.lambda_structure*torch.max(threshold,structure_loss)

        loss = self.lambda_text*text_loss + thresholded_structure_loss

        
        return loss, gen_images, prompt_c
            
            