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
         eval_dataloader,
         times,
         weathers,
         save_image_path:str,
         epoch,
         origin_alpha,
         device,
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
             time_condition_number,
             weather_condition_number,
             real_image_tensor,
             image_embedding,
             model_time_input_embedding,
             model_weather_input_embedding,
             latents,
             time_sd_text_embedding,
             weather_sd_text_embedding,
             from_clip_embedding,
             to_time_clip_embedding,
             to_weather_clip_embedding)= inputs
            
            idx = idx.numpy()
            selected_time_conditions = [times[i] for i in time_condition_number]
            selected_weather_conditions = [weathers[i] for i in weather_condition_number]
            real_image_tensor=real_image_tensor.to(device)
            image_embedding=image_embedding.to(device)
            #model_time_input_embedding = model_time_input_embedding.to(device)
            #model_weather_input_embedding = model_weather_input_embedding.to(device)
            latents = latents.to(device)
            time_sd_text_embedding = time_sd_text_embedding.to(device)
            weather_sd_text_embedding = weather_sd_text_embedding.to(device)
            from_clip_embedding=from_clip_embedding.to(device)
            to_time_clip_embedding = to_time_clip_embedding.to(device)
            to_weather_clip_embedding = to_weather_clip_embedding.to(device)
                   
            model_input = torch.cat([image_embedding,
                                        #from_clip_embedding,
                                        to_time_clip_embedding,
                                        to_weather_clip_embedding], dim=1).view(len(idx), 3, -1)
            
            pred_ginit, pred_portion = model(model_input)

            to_clip_embedding = torch.cat([to_time_clip_embedding, to_weather_clip_embedding])
            sd_text_embedding = torch.cat([time_sd_text_embedding, weather_sd_text_embedding])
            
            
            loss, gen_images, _ccs, _dcs  = criterion(real_image_tensor=real_image_tensor,
                                                      clip_real_image_embedding=image_embedding,
                                                      from_clip_embedding=from_clip_embedding,
                                                      to_clip_embedding=to_clip_embedding,
                                                      model_input_embedding=None,
                                                      image_latents=latents,
                                                      sd_prompt_embedding=sd_text_embedding,
                                                      g_init=pred_ginit,
                                                      origin_alpha=origin_alpha,
                                                      g_portion=pred_portion)
            t.set_postfix(loss=loss.item())
            
            time_ccs, weather_ccs = _ccs.chunk(2)
            predicts = pred_ginit*pred_portion.squeeze()
            preds = predicts.detach().cpu().numpy()
            weather_ccs = weather_ccs.detach().cpu().numpy()
            time_ccs = time_ccs.detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            
            real_imgs = [T.ToPILImage()(latent) for latent in real_image_tensor]
            edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
            for a in range(len(edited_imgs)):
                str_w_ccs = f'{weather_ccs.item(a):.2f}'.replace('.','_')
                str_t_ccs = f'{time_ccs.item(a):.2f}'.replace('.','_')
                str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                s = save_dir/f'{idx.item(a):04}-{selected_time_conditions[a]}-{int(preds.item(a,0))}-{str_t_ccs}-{selected_weather_conditions[a]}-{int(preds.item(a,1))}-{str_w_ccs}-{str_dcs}.png'
                edited_imgs[a].save(s)
                save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
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
                                         #from_clip_embedding,
                                         to_clip_embedding], dim=1).view(len(idx), 2, -1)
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
            ccs = _ccs.detach().cpu().numpy()
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
        
        

@torch.no_grad()
def blip_eval(model,
            criterion,
            eval_dataloader,
            eval_prompts,
            domains,
            save_image_path:str,
            epoch,
            length,
            device,
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
             prompt_idx,
             condition_number,
             real_image_tensor,
             image_embedding,
             condition_mean,
             latents,
             sd_text_embedding,
             to_clip_embedding,
             blip_sd_embedding,
             blip_clip_embedding)= inputs
            
            idx = idx.numpy()
            prompt_idx = prompt_idx.numpy()
            selected_conditions = [domains[i] for i in condition_number]
            
            selected_prompts = []
            for j, con in enumerate(selected_conditions):
                condition_prompts = list(eval_prompts[con]['prompt_emb_pair'].keys())
                selected_prompts.append(condition_prompts[prompt_idx[j]])
                
            real_image_tensor=real_image_tensor.to(device)
            image_embedding=image_embedding.to(device)
            #condition_mean = condition_mean.to(device)
            latents = latents.to(device)
            sd_text_embedding = sd_text_embedding.to(device)
            to_clip_embedding = to_clip_embedding.to(device)
            blip_sd_embedding = blip_sd_embedding.to(device)
            blip_clip_embedding = blip_clip_embedding.to(device)
            

            model_input = torch.cat([image_embedding,
                                        blip_clip_embedding,
                                        to_clip_embedding], dim=1).view(len(idx), length, -1)

            pred_blip_ginit, pred_ginit = model(model_input)
            total_g = pred_blip_ginit+pred_ginit
            portion1 = pred_blip_ginit/total_g
            portion2 = 1-portion1
            portion = torch.cat([portion1, portion2], dim=1)
            
            sd_text_embedding = torch.cat([blip_sd_embedding, sd_text_embedding])
            
            loss, gen_images, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                from_clip_embedding=blip_clip_embedding,
                                                to_clip_embedding=to_clip_embedding,
                                                image_latents=latents,
                                                sd_prompt_embedding=sd_text_embedding,
                                                g_init=total_g,
                                                g_portion=portion)
            t.set_postfix(loss=loss.item())
            
            blip_ccs, condition_ccs = _ccs.chunk(2)
            predicts = total_g*portion.squeeze()
            preds = predicts.detach().cpu().numpy()
            condition_ccs = condition_ccs.detach().cpu().numpy()
            blip_ccs = blip_ccs.detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            
            real_imgs = [T.ToPILImage()(latent) for latent in real_image_tensor]
            edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
            for a in range(len(edited_imgs)):
                str_w_ccs = f'{condition_ccs.item(a):.2f}'.replace('.','_')
                str_t_ccs = f'{blip_ccs.item(a):.2f}'.replace('.','_')
                str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                s = save_dir/f'{idx.item(a):04}-{int(preds.item(a,0))}-{str_t_ccs}-{selected_conditions[a]}-{selected_prompts[a]}-{int(preds.item(a,1))}-{str_w_ccs}-{str_dcs}.png'
                edited_imgs[a].save(s)
                save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
    return epoch_loss   

@torch.no_grad()
def zero_shot_eval(model,
                   criterion,
                   eval_dataloader,
                   eval_prompts,
                   domains,
                   save_image_path:str,
                   epoch,
                   length,
                   device,
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
             prompt_idx,
             condition_number,
             real_image_tensor,
             image_embedding,
             condition_mean,
             latents,
             sd_text_embedding,
             to_clip_embedding)= inputs
        
            idx = idx.numpy()
            prompt_idx = prompt_idx.numpy()
            selected_conditions = [domains[i] for i in condition_number]
            selected_prompts = []
            for j, con in enumerate(selected_conditions):
                condition_prompts = list(eval_prompts[con]['prompt_emb_pair'].keys())
                selected_prompts.append(condition_prompts[prompt_idx[j]])
                
            real_image_tensor=real_image_tensor.to(device)
            image_embedding=image_embedding.to(device)
            condition_mean = condition_mean.to(device)
            latents = latents.to(device)
            sd_text_embedding = sd_text_embedding.to(device)
            to_clip_embedding = to_clip_embedding.to(device)

            model_input = torch.cat([image_embedding,
                                     #condition_mean,
                                     to_clip_embedding], dim=1).view(len(idx), length, -1)

            pred_ginit = model(model_input)
            
            
            loss, gen_images, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                     condition_mean=condition_mean,
                                                     image_latents=latents,
                                                     sd_prompt_embedding=sd_text_embedding,
                                                     to_clip_embedding=to_clip_embedding,
                                                     g_init=pred_ginit)
            t.set_postfix(loss=loss.item())
            
            mean_ccs, condition_ccs = _ccs.chunk(2)
            preds = pred_ginit.squeeze().detach().cpu().numpy()
            condition_ccs = condition_ccs.detach().cpu().numpy()
            mean_ccs = mean_ccs.detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            
            if i%2==0:
                real_imgs = [T.ToPILImage()(latent) for latent in real_image_tensor]
                edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
                for a in range(len(edited_imgs)):
                    str_w_ccs = f'{condition_ccs.item(a):.2f}'.replace('.','_')
                    str_m_ccs = f'{mean_ccs.item(a):.2f}'.replace('.','_')
                    str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                    s = save_dir/f'{idx.item(a):04}-{int(preds.item(a))}-{selected_prompts[a]}-{selected_conditions[a]}-{str_w_ccs}-{str_m_ccs}-{str_dcs}.png'
                    edited_imgs[a].save(s)
                    save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                    real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
    return epoch_loss   