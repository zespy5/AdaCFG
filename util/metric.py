from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
dino_model = AutoModel.from_pretrained('facebook/dinov2-large').to('cuda')

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

cosin_sim = nn.CosineSimilarity(dim=0)
imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
global_resize_transform = T.Resize(224, max_size=480)
totensor = T.ToTensor()


@torch.no_grad()
def Clip(image, prompt):
    
    clip_inputs = clip_processor(text=prompt,
                                 images= image,
                                 return_tensors="pt",
                                 padding=True)
    
    texts = clip_inputs['input_ids'].to('cuda')
    attn_mask = clip_inputs['attention_mask'].to('cuda')
    images = clip_inputs['pixel_values'].to('cuda')
    text_feautre = clip_model.get_text_features(texts, attn_mask)
    image_features = clip_model.get_image_features(images)

    
    sim = F.cosine_similarity(text_feautre, image_features, dim=1).item()

    return sim
@torch.no_grad()
def Clip_ds(real_image, gen_image, from_prompt, to_prompt):
    
    clip_inputs = clip_processor(text=[from_prompt, to_prompt],
                                 images= [real_image,gen_image],
                                 return_tensors="pt",
                                 padding=True)
    
    texts = clip_inputs['input_ids'].to('cuda')
    attn_mask = clip_inputs['attention_mask'].to('cuda')
    images = clip_inputs['pixel_values'].to('cuda')
    text_feautre = clip_model.get_text_features(texts, attn_mask)
    image_features = clip_model.get_image_features(images)

    text1, text2 = text_feautre.chunk(2)
    image1,image2 = image_features.chunk(2)
    
    text_direction = text2 - text1
    image_direction = image2 - image1
    
    sim = F.cosine_similarity(text_direction, image_direction, dim=1).item()
    
    return sim

@torch.no_grad()
def Clip_ds2(conditions, real_image, gen_image, from_prompt, to_prompt):
    
    prompt_candidates = [from_prompt+con for con in conditions]
    prompt_candidates.append(to_prompt)

    clip_inputs = clip_processor(text=prompt_candidates,
                                 images= [real_image,gen_image],
                                 return_tensors="pt",
                                 padding=True)
    
    text = clip_inputs['input_ids'].to('cuda')
    attn_mask = clip_inputs['attention_mask'].to('cuda')
    images = clip_inputs['pixel_values'].to('cuda')
    text_features = clip_model.get_text_features(text, attn_mask)
    image_features = clip_model.get_image_features(images)

    to_text_features = text_features[-1].unsqueeze(0)
    candidates = text_features[:-1]

    image1,image2 = image_features.chunk(2)
    
    image_direction = image2 - image1
    
    from_image = image1.repeat(len(conditions),1)

    cos_sim = F.cosine_similarity(candidates, from_image, dim=1)
    max_idx = torch.argmax(cos_sim)
    from_text_feature = candidates[max_idx].unsqueeze(0)
    text_direction = to_text_features- from_text_feature

    sim = F.cosine_similarity(text_direction, image_direction, dim=1).item()

    return sim, max_idx.item()
  
@torch.no_grad()
def Dino(real, gen):
    
    images = [real, gen]

    inputs = dino_processor(images=images, return_tensors="pt").to('cuda')
    outputs = dino_model(**inputs)
    image_features = outputs.last_hidden_state
    image_features1,image_features2 = image_features.view(2,-1).chunk(2)
        
    sim = F.cosine_similarity(image_features1, image_features2, dim=1).item()

    return sim

    
@torch.no_grad()
def Clip_sturcture(real, gen):
    
    clip_inputs = clip_processor(images= [real,gen],
                                 return_tensors="pt",
                                 padding=True)

    images = clip_inputs['pixel_values'].to('cuda')
    outputs = clip_model.vision_model(images)[0]
    
    real_feature, gen_feature = outputs.view(2,-1).chunk(2)
    
    sim = F.cosine_similarity(real_feature, gen_feature, dim=1).item()

    return sim