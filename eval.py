import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from util.loss import Loss
from util.utils import *
from models.model import GuidanceModel
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from random import randint
@torch.no_grad()
def eval(model,
         criterion,
         data_root:str,
         conditions,
         save_image_path:str,
         epoch,
         device,
         latents_save_root='eval_latents_forward'):
    num_condition = len(conditions)
    data_root = Path(data_root)
    eval_datas = sorted([*data_root.glob('*')])
    
    seasons, weathers, times = conditions
    
    
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    generate_prompt = criterion.pipeline.generate_prompt
    
    save_root = Path(save_image_path)
    save_root.mkdir(exist_ok=True, parents=True)
    save_dir = save_root/f'epoch-{epoch}'
    save_dir.mkdir(exist_ok=True)
    
    num_instance = len(eval_datas)
    batch_size = 5
    total_loss = 0
    print('Evaluate')
    with tqdm(eval_datas) as t:
        for i, image_path in enumerate(t):
            #image to text
            image_dirs = [image_path]
            real_images = [Image.open(image_path).convert('RGB')]
            original_prompts = [generate_prompt(real_images[0])]
            
            image_dirs = image_dirs*batch_size
            real_images = real_images*batch_size
            original_prompts = original_prompts*batch_size
            
            #select random condition

            ##construct_prompts = [p.replace(' at night','') for p in original_prompts]
            ##conditioned_prompts = [construct_prompts[i]+conditions[i] for i in range(num_conditions)]

            #text clip embedding : model inputs
            ##original_image_emb = original_image_embedds(real_images)
            ##conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
            ##prompt_emb = torch.cat([original_image_emb, conditioned_prompt_emb], dim=1)
            season_prompts = [seasons[randint(0,3)] for _ in range(batch_size)]
            weather_prompts = [weathers[randint(0,4)] for _ in range(batch_size)]
            time_prompts = [times[randint(0,3)] for _ in range(batch_size)]
            
            s_season_prompts = [original_prompts[i]+season_prompts[i] for i in range(batch_size)]
            s_weather_prompts = [original_prompts[i]+weather_prompts[i] for i in range(batch_size)]
            s_time_prompts = [original_prompts[i]+time_prompts[i] for i in range(batch_size)]
            season_prompt_embed = conditioned_prompt_embedds(s_season_prompts).unsqueeze(1)
            weather_prompt_embed = conditioned_prompt_embedds(s_weather_prompts).unsqueeze(1)
            time_prompt_embed = conditioned_prompt_embedds(s_time_prompts).unsqueeze(1)
            
            style_prompts = [season_prompts, weather_prompts, time_prompts]

            original_image_emb = original_image_embedds(real_images).unsqueeze(1)

            prompt_emb = torch.cat([original_image_emb,
                                    season_prompt_embed,
                                    weather_prompt_embed,
                                    time_prompt_embed], dim=1)
            prompt_emb = prompt_emb.to(device)
            
            pred_ginit, pred_gportion = model(prompt_emb)
            pred_ginit = pred_ginit.to(device)
            pred_gportion = pred_gportion.to(device)
            
            loss, gen_images, prompts_c, _ccs, _dcs = criterion(image_dirs=image_dirs,
                                                    real_images=real_images,
                                                    prompts=style_prompts, 
                                                    g_init=pred_ginit,
                                                    g_portion= pred_gportion,
                                                    latents_save_root= latents_save_root)
            t.set_postfix(loss=loss.item())
            
            predicts = pred_ginit*pred_gportion.squeeze()
            preds = predicts.detach().cpu().numpy()
            season_ccs = _ccs[0].detach().cpu().numpy()
            weather_ccs = _ccs[1].detach().cpu().numpy()
            time_ccs = _ccs[2].detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
            for a in range(len(edited_imgs)):
                length = len([*save_dir.glob('*')])
                str_s_ccs = f'{season_ccs.item(a):.2f}'.replace('.','_')
                str_w_ccs = f'{weather_ccs.item(a):.2f}'.replace('.','_')
                str_t_ccs = f'{time_ccs.item(a):.2f}'.replace('.','_')
                str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                s = save_dir/f'{length:03}-{prompts_c[0][a]}-{int(preds.item(a,0))}-{str_s_ccs}-{style_prompts[1][a]}-{int(preds.item(a,1))}-{str_w_ccs}-{style_prompts[2][a]}-{int(preds.item(a,2))}-{str_t_ccs}-{str_dcs}.png'
                edited_imgs[a].save(s)

            total_loss += loss.item()
        epoch_loss = total_loss/(num_instance)
        
    return epoch_loss

@torch.no_grad()
def linear_eval(model,
                criterion,
                eval_dataloader,
                conditions,
                epoch,
                save_image_path:str,
                device,
                origin_alpha,
                model_class
                ):
    
    total_loss = 0

    save_root = Path(save_image_path)
    save_root.mkdir(exist_ok=True, parents=True)
    save_dir = save_root/f'epoch-{epoch}'
    save_dir.mkdir(exist_ok=True)
    
    
    print('Evaluate')
    with tqdm(eval_dataloader) as t:
        for i, inputs in enumerate(t):
            
            (idx,
             condition_number,
             real_image_tensor,
             image_embedding,
             model_input_embedding,
             latents,
             sd_text_embedding,
             from_clip_embedding,
             to_clip_embedding)= inputs
            #image to text
            idx = idx.numpy()
            selected_conditions = [conditions[i] for i in condition_number]
            real_image_tensor=real_image_tensor.to(device)
            image_embedding=image_embedding.to(device)
            model_input_embedding = model_input_embedding.to(device)
            latents = latents.to(device)
            sd_text_embedding = sd_text_embedding.to(device)
            from_clip_embedding=from_clip_embedding.to(device)
            to_clip_embedding = to_clip_embedding.to(device)

            if model_class:
                model_input = torch.cat([image_embedding,
                                         from_clip_embedding,
                                         to_clip_embedding], dim=1).view(len(idx), 3, -1)
            else:
                model_input = torch.cat([image_embedding,
                                         from_clip_embedding,
                                         to_clip_embedding], dim=1)
                
                
            pred_ginit = model(model_input)

            loss, gen_images, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                     clip_real_image_embedding=image_embedding,
                                                     from_clip_embedding=from_clip_embedding,
                                                     to_clip_embedding=to_clip_embedding,
                                                     model_input_embedding=model_input_embedding,
                                                     image_latents=latents,
                                                     sd_prompt_embedding=sd_text_embedding,
                                                     g_init=pred_ginit,
                                                     origin_alpha=origin_alpha,
                                                     g_portion=None)
            t.set_postfix(loss=loss.item())
            
            preds = pred_ginit.squeeze().detach().cpu().numpy()
            ccs = _ccs[0].detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            
            real_imgs = [T.ToPILImage()(latent) for latent in real_image_tensor]
            edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
            for a in range(len(edited_imgs)):
                str_s_ccs = f'{ccs.item(a):.2f}'.replace('.','_')
                str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                s = save_dir/f'{idx.item(a):04}-{selected_conditions[a]}-{int(preds.item(a))}-{str_s_ccs}-{str_dcs}.png'
                edited_imgs[a].save(s)
                save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
    return epoch_loss
        
        