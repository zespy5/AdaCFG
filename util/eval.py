import torch
from util.utils import *
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as T


@torch.no_grad()
def ip2p_eval(model,
                   criterion,
                   eval_dataloader,
                   eval_prompts,
                   domains,
                   save_image_path:str,
                   epoch,
                   config,
                   device
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
            sd_text_embedding = sd_text_embedding.to(device)
            to_clip_embedding = to_clip_embedding.to(device)

            if config['Attention']:
                model_input = torch.cat([image_embedding,
                                            to_clip_embedding], dim=1).view(len(idx), config['model']['length'], -1)
            else:
                model_input = torch.cat([image_embedding,
                                            to_clip_embedding], dim=1)
                    
            pred_ginit, pred_velocity = model(model_input)
            
            
            loss, gen_images, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                     condition_mean=condition_mean,
                                                     sd_prompt_embedding=sd_text_embedding,
                                                     to_clip_embedding=to_clip_embedding,
                                                     g_init=pred_ginit,
                                                     velocity=pred_velocity)
            t.set_postfix(loss=loss.item())
            
            mean_ccs, condition_ccs = _ccs.chunk(2)
            preds = pred_ginit.squeeze().detach().cpu().numpy()
            preds_v = pred_velocity.squeeze().detach().cpu().numpy()
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
                    str_v = f'{preds_v.item(a):.2f}'.replace('.','_')
                    str_g = f'{preds.item(a):.2f}'.replace('.','_')
                    s = save_dir/f'{idx.item(a):04}-{str_g}-{str_v}-{selected_prompts[a]}-{selected_conditions[a]}-{str_w_ccs}-{str_m_ccs}-{str_dcs}.png'
                    edited_imgs[a].save(s)
                    save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                    real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
    return epoch_loss   

@torch.no_grad()
def eval(model,
                   criterion,
                   eval_dataloader,
                   eval_prompts,
                   domains,
                   save_image_path:str,
                   epoch,
                   device
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
                                     to_clip_embedding], dim=1)
                    
            pred_ginit, pred_velocity = model(model_input)
            
            
            loss, gen_images, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                     condition_mean=condition_mean,
                                                     image_latents=latents,
                                                     sd_prompt_embedding=sd_text_embedding,
                                                     to_clip_embedding=to_clip_embedding,
                                                     g_init=pred_ginit,
                                                     velocity=pred_velocity)
            t.set_postfix(loss=loss.item())
            

            preds = pred_ginit.squeeze().detach().cpu().numpy()
            preds_v = pred_velocity.squeeze().detach().cpu().numpy()
            condition_ccs = _ccs.detach().cpu().numpy()
            struc_dcs = _dcs.detach().cpu().numpy()
            
            if i%2==0:
                #real_imgs = [T.ToPILImage()(latent) for latent in real_image_tensor]
                edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
                for a in range(len(edited_imgs)):
                    str_w_ccs = f'{condition_ccs.item(a):.2f}'.replace('.','_')
                    str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                    str_v = f'{preds_v.item(a):.2f}'.replace('.','_')
                    str_g = f'{preds.item(a):.2f}'.replace('.','_')
                    s = save_dir/f'{idx.item(a):04}-{str_g}-{str_v}-{selected_prompts[a]}-{selected_conditions[a]}-{str_w_ccs}-{str_dcs}.png'
                    edited_imgs[a].save(s)
                    #save_origin_img = save_dir/f'{idx.item(a):04}-real_image.png'
                    #real_imgs[a].save(save_origin_img)

            total_loss += loss.item()
        epoch_loss = total_loss/len(eval_dataloader)
        
    return epoch_loss   
