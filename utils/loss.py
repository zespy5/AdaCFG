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
                 conditions : Optional[Union[str, List[str]]]= None,
                 negative_prompt : str = 'ugly, blurry, low res, unrealistic, paint',
                 lambda_text : float = 0.5,
                 lambda_structure: float = 2.5,
                 device :str = 'cuda',
                 data_root : str='image_data',
                 ):
        super().__init__()
        
        self.device = device
        self.conditions = conditions
        self.negative_prompt = negative_prompt
        self.lambda_text = lambda_text
        self.lambda_structure = lambda_structure
    
        
        self.data_root = Path(data_root)
        
        self.pipeline = PnPPipeline(generate_condition_prompt=False,
                                    device=self.device,
                                    tensor_out=True)
        self.guidance_scheduler = GuidanceScheduler()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

        for p in self.clip_model.parameters():
            p.requires_grad=False
        
        for p in self.dino_model.parameters():
            p.requires_grad=False
    #@torch.no_grad()
    def prompt_embeds(self, prompts):
        clip_inputs = self.clip_processor(text=prompts,
                                            return_tensors="pt", 
                                            padding=True)
        text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                            clip_inputs["attention_mask"].to(self.device))
            
        return text_features
        
    def generate_edited_image(self, image_dirs, prompts, guidance_info):

        scheduled_guidance = self.guidance_scheduler.get_guidance_scales(guidance_info)
    
        outputs = self.pipeline(image_dirs=image_dirs,
                                prompts = prompts,
                                guidance_scales=scheduled_guidance,
                                negative_prompt=self.negative_prompt)
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
            
    def clip_loss(self, gen_images, prompts, eps = 1e-8):
        
        with torch.no_grad():
            clip_inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(clip_inputs["input_ids"].to(self.device),
                                                              clip_inputs["attention_mask"].to(self.device))
        

        transform_gen_images = self.clip_transform()(gen_images)
        img_features = self.clip_model.get_image_features(transform_gen_images.to(self.device))

        #loss = F.mse_loss(img_features, text_features)
        loss = F.cosine_similarity(text_features, img_features, dim=1)+eps
        loss = loss.view(-1).mean()
        return 1/loss
    
    def dino_loss(self, real_images, gen_images):
        
        with torch.no_grad():
            real_inputs = self.dino_processor(images=real_images, return_tensors="pt").to(self.device)
            real_outputs = self.dino_model(**real_inputs).last_hidden_state

        gen_inputs = self.dino_transform()(gen_images).to(self.device)
        gen_outputs = self.dino_model(gen_inputs).last_hidden_state
        
        loss = F.mse_loss(real_outputs, gen_outputs)
        
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

        
        return loss, gen_images
            
            