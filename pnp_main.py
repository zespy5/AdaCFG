from util.guidance_scheduler import GuidanceScheduler
from util.pnp import PnPPipeline
from models.model import *
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from util.utils import get_json, get_config
from util.metric import Clip, Dino
import argparse

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
def get_target_domain(condition_config, target_prompt):
    means = []
    for prompts in condition_config.values():
        mean_feautre = make_mean(prompts)
        means.append(mean_feautre)
        
    keys=list(condition_config.keys())
    
    means_tensor = torch.cat(means)
        
    prompt_emb = prompt_embeds(target_prompt)

    
    from_image = prompt_emb.repeat(len(means_tensor),1)

    cos_sim = F.cosine_similarity(means_tensor, from_image, dim=1)
    max_idx = torch.argmax(cos_sim)

    return keys[max_idx]

@torch.no_grad()
def main(args):
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    config_setting = get_config(args.model_config)
    model_setting = config_setting['model']
    model_setting['hidden_dim'] *=2
    base_setting = config_setting['loss']

    model_path = Path(args.model_path)
    
    model = GuidanceModel(**model_setting).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    prompt = args.prompt
    augmented_prompts = get_json(args.augmented_prompts)
    target_domain = get_target_domain(augmented_prompts, prompt)
    
    candidates = augmented_prompts[target_domain]
    
    batch_size = len(candidates)
    
    
    pnp_attn_t=base_setting['pnp_injection_rate']
    pnp_f_t=base_setting['pnp_res_injection_rate']
    n_timestep=base_setting['n_timestep']
    latents_steps=base_setting['latents_steps']
    gradient = base_setting['gradient']

    pnp = PnPPipeline(pnp_attn_t=pnp_attn_t,
                      pnp_f_t=pnp_f_t,
                      n_timestep=n_timestep,
                      latents_steps=latents_steps)
    
    guidance_scheduler = GuidanceScheduler(n_timestep=n_timestep,
                                           gradient=gradient)
    
    negative_prompt = args.negative_prompt
    image_path = Path(args.image_path)
    image = Image.open(image_path).resize((512,512))
    
    
    image_embedding = image_clip_embeds(image)
    to_clip_embedding = prompt_embeds(candidates)
    img_embs = image_embedding.repeat(batch_size,1)
    model_input = torch.cat([img_embs, to_clip_embedding], dim=1)
    guidance_value, velocity = model(model_input)

    guidance = guidance_scheduler.get_guidance_scales(guidance_value, velocity)

    images = [image_path]*batch_size
    outputs = pnp(image_dirs=images,
                  negative_prompt=negative_prompt,
                  prompts=candidates,
                  guidance_scales=guidance,
                  latents_save_root='inference_test_latents')

    gen_images = outputs.images

    losses = []
    for i in range(batch_size):

        p_clip = Clip(gen_images[i], prompt)
        n_clip = Clip(gen_images[i], negative_prompt)
        dino = Dino(image, gen_images[i])
        loss = p_clip + (1-n_clip) + dino*0.1
        losses.append(loss)

    max_loss = max(losses)
    max_index = losses.index(max_loss)
    
    max_image = gen_images[max_index]
    save_image = save_dir/f'{prompt}.png'
    max_image.save(save_image) 
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/config.yaml"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of a street at night."
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--augmented_prompts",
        type=str,
        default="configs/conditions.json"
    )
    
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default = "ugly, low resolution, unrealistic, distortion, blurry"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        required=True
    )
    
    args= parser.parse_args()
    
    main(args)
    
    
    
    
    