import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from utils.loss import Loss
from utils.utils import *
from models.model import GuidanceModel
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from utils.utils import generate_prompt

@torch.no_grad()
def eval(model,
         data_root:str,
         conditions,
         save_image_path:str,
         epoch,
         device,
         model_device):
    data_root = Path(data_root)
    eval_datas = [*data_root.glob('*')]
    criterion = Loss(lambda_text=2.5,
                     lambda_structure=1.0,
                     device=device,
                     data_root=data_root,
                     latents_save_root='eval_latents_forward').to(device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    
    save_root = Path(save_image_path)
    save_root.mkdir(exist_ok=True, parents=True)
    save_dir = save_root/f'epoch-{epoch}'
    save_dir.mkdir(exist_ok=True)
    
    num_instance = len(eval_datas)
    num_conditions = len(conditions)
    total_loss = 0
    print('Evaluate')
    with tqdm(eval_datas) as t:
        for i, image_path in enumerate(t):
            #image to text
            image_dirs = [image_path]
            real_images = [Image.open(image_path).convert('RGB')]
            original_prompts = [generate_prompt(real_images[0])]
            
            image_dirs = image_dirs*num_conditions
            real_images = real_images*num_conditions
            original_prompts = original_prompts*num_conditions
            
            #select random condition

            construct_prompts = [p.replace(' at night','') for p in original_prompts]
            conditioned_prompts = [construct_prompts[i]+conditions[i] for i in range(num_conditions)]

            #text clip embedding : model inputs
            original_image_emb = original_image_embedds(real_images)
            conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
            prompt_emb = torch.cat([original_image_emb, conditioned_prompt_emb], dim=1)
            prompt_emb = prompt_emb.to(model_device)
            
            predicts = model(prompt_emb)
            predicts = predicts.to(device)
            
            
            loss, gen_images, prompts_c = criterion(image_dirs=image_dirs,
                                    real_images=real_images,
                                    prompts=conditioned_prompts, 
                                    guidance_info= predicts)
            t.set_postfix(loss=loss.item())
            
            preds = predicts.detach().cpu().numpy()
            edited_imgs = [T.ToPILImage()(latent) for latent in gen_images]
            for a in range(len(edited_imgs)):
                length = len([*save_dir.glob('*')])
                s = save_dir/f'{length:03}-{prompts_c[a]}-{int(preds.item(a,0))}.png'
                edited_imgs[a].save(s)

            total_loss += loss.item()
        epoch_loss = total_loss/(num_instance)
        
    return epoch_loss
        
        