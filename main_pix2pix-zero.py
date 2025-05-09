import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler
from pix2pixzero_hack import Pix2PixZeroPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import json
from models.model import *
from util.guidance_scheduler import GuidanceScheduler
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@torch.no_grad()
def prompt_embeds(prompts):
    clip_inputs = clip_processor(text=prompts,
                                        return_tensors="pt", 
                                        padding=True)
    text_features = clip_model.get_text_features(clip_inputs["input_ids"].to('cuda'),
                                                        clip_inputs["attention_mask"].to('cuda'))
        
    return text_features

@torch.no_grad()
def image_clip_embeds(image):
    clip_inputs = clip_processor(images=image,
                                        return_tensors="pt", 
                                        padding=True)
    image_features = clip_model.get_image_features(clip_inputs["pixel_values"].to('cuda'))
        
    return image_features

def get_json(config_path):
    with open(config_path, "r",encoding="euc-kr", errors="replace") as f:
        config = json.load(f)
        
    return config

@torch.no_grad()
def make_mean(conditions):
    clip_inputs = clip_processor(text=conditions,
                                 return_tensors="pt",
                                 padding=True)
    
    text = clip_inputs['input_ids'].to('cuda')
    attn_mask = clip_inputs['attention_mask'].to('cuda')
    text_features = clip_model.get_text_features(text, attn_mask)
    mean_feature = text_features.mean(dim=0)[None]

    return mean_feature


@torch.no_grad()
def get_origin_domain(condition_config, real_image):
    means = []
    for prompts in condition_config.values():
        mean_feautre = make_mean(prompts)
        means.append(mean_feautre)
        
    keys=list(condition_config.keys())
    
    means_tensor = torch.cat(means)
        
    clip_inputs = clip_processor(images=real_image,
                                 return_tensors="pt",
                                 padding=True)
    
    images = clip_inputs['pixel_values'].to('cuda')
    image_features = clip_model.get_image_features(images)

    
    from_image = image_features.repeat(len(means_tensor),1)

    cos_sim = F.cosine_similarity(means_tensor, from_image, dim=1)
    max_idx = torch.argmax(cos_sim)

    return keys[max_idx]

name_dict = {"clear":'a photo of a street on a clear day',
             "cloudy":'a photo of a street on a cloudy day',
             "fog":'a photo of a street on a foggy day',
             "rain":'a photo of a street on a rainy day',
             "snow":'a photo of a street on a snowy day',
             "night":'a photo of a street at night',
             "sunset":'a photo of a street at sunset'}

condition_config = get_json('configs/large_conditions.json')
categorys = list(condition_config.keys())


captioner_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(captioner_id)
model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

sd_model_ckpt = "stabilityai/stable-diffusion-2-1-base"
pipeline = Pix2PixZeroPipeline.from_pretrained(
    sd_model_ckpt,
    caption_generator=model,
    caption_processor=processor,
    safety_checker=None,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

num_inference = 100
guidance_scheduler = GuidanceScheduler(gradient='decrease', device='cuda', n_timestep=num_inference)

model_path = Path('ckpts/best_ckpts/15_linear_weather.pt')

model = GuidanceModel(init_g=50.0,
                        divide_out=0.1,
                        hidden_dim=768*2,
                        num_layers=5,
                        num_guidance_info=1).to('cuda')

model.load_state_dict(torch.load(model_path))
model.eval()

data_root = Path('image_data/eval')
origin_images = sorted([*data_root.glob('*.jpg')])
save_root = Path('check_valid/pix2pix-zero')
save_root.mkdir(exist_ok=True)
for c in categorys:
    category_root = save_root/name_dict[c]
    category_root.mkdir(exist_ok=True)

with torch.no_grad():
    for o_img in origin_images:
        init_image = Image.open(o_img).convert("RGB")
        origin_domain = get_origin_domain(condition_config,init_image)
        images = [init_image]*len(categorys)
        
        caption = pipeline.generate_caption(init_image)
        source_embeds = pipeline.get_embeds(condition_config[origin_domain])
        generator = torch.manual_seed(0)
        inv_latents = pipeline.invert(caption, 
                                    image=init_image, 
                                    generator=generator,
                                    num_inference_steps=num_inference).latents

        image_embedding = image_clip_embeds(init_image)
            
        for k,v in condition_config.items():
            target_prompt = name_dict[k]
            to_clip_embedding = prompt_embeds(target_prompt)
            
            target_prompts = v
            target_embeds = pipeline.get_embeds(target_prompts)
            
            model_input = torch.cat([image_embedding, to_clip_embedding], dim=1)
            guidance_value = model(model_input)
            guidance = guidance_scheduler.get_guidance_scales(guidance_value).to('cuda')

            image = pipeline(
                caption,
                source_embeds=source_embeds,
                target_embeds=target_embeds,
                num_inference_steps=num_inference,
                cross_attention_guidance_amount=0.05,
                generator=generator,
                latents=inv_latents,
                negative_prompt=caption,
                guidance_scale=guidance,
            ).images[0]
            save_name = save_root/name_dict[k]/f'{o_img.stem}.png'
            image.save(save_name)

