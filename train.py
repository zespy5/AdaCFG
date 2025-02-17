import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from util.loss import Loss
from util.utils import *
from models.model import GuidanceModel, AttentionModel
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from eval import eval
from PIL import Image
from random import randint



def train(data_root, train_device):
    timestamp = get_timestamp()
    
    name = f"work-{timestamp}-tranformer struc+condition 3,pnp0.9 lambda_t 1, lambda_s 1, dino squre loss thres 0.3, init 200, lr 0.0001"
        ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        name=name,
        mode=os.environ.get("WANDB_MODE"),
    )
    
    seed_everything(54)
    
    seasons = [' on a summer day',
               ' on a spring day',
               ' on a winter day',
               ' on a autumn day',
               '']
    
    weathers = [' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a sunny day',
                ' on a cloudy day',
                '']
    
    times = [' at night',
             ' at sunset',
             ' at sunrise',
             ' at daytime',
             '']
    
    conditions = [seasons, weathers, times]
    
    batch_size = 5
    
    dataset = DomainChangeDataset(data_directory=data_root)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                          
    model = AttentionModel(hidden_dim=768,
                           heads=4,
                           init_g=200.0).to(train_device)
    
    criterion = Loss(lambda_text=1.0,
                     lambda_structure=1.0,
                     device=train_device,
                     data_root=data_root,
                     dino_threshold=0.4,
                     num_condition=3,
                     generate_condition_prompt=True,
                     pnp_injection_rate=0.9).to(train_device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    generate_prompt = criterion.pipeline.generate_prompt
    lr = 0.0001

    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                            lr_lambda=lambda epoch: 0.99*epoch)
    min_val_loss = 1000
    
    for epoch in range(50):
        print(f"epoch : {epoch}")
        #dataset.update_condition_set()
        total_loss = 0
        with tqdm(dataloader) as t:
            for k, image_idx in enumerate(t):
                #image to text
                idxs= image_idx.numpy()
                image_dirs = [data_root/f'{idx:03}.png' for idx in idxs]
                real_images = [Image.open(img).convert('RGB') for img in image_dirs]
                original_prompts = generate_prompt(real_images)
                
                data_len = len(image_dirs)
                season_prompts = [seasons[randint(0,4)] for _ in range(data_len)]
                weather_prompts = [weathers[randint(0,5)] for _ in range(data_len)]
                time_prompts = [times[randint(0,4)] for _ in range(data_len)]
                
                s_season_prompts = [original_prompts[i]+season_prompts[i] for i in range(data_len)]
                s_weather_prompts = [original_prompts[i]+weather_prompts[i] for i in range(data_len)]
                s_time_prompts = [original_prompts[i]+time_prompts[i] for i in range(data_len)]
                season_prompt_embed = conditioned_prompt_embedds(s_season_prompts).unsqueeze(1)
                weather_prompt_embed = conditioned_prompt_embedds(s_weather_prompts).unsqueeze(1)
                time_prompt_embed = conditioned_prompt_embedds(s_time_prompts).unsqueeze(1)
                
                style_prompts = [season_prompts, weather_prompts, time_prompts]
                
                #make conditioned prompt
                ##construct_prompts = [p.replace(' at night','') for p in original_prompts]
                ##conditioned_prompts = [construct_prompts[i]+style_prompts[i] for i in range(len(idxs))]
                
                #text clip embedding : model inputs
                original_image_emb = original_image_embedds(real_images).unsqueeze(1)
                ##conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
                prompt_emb = torch.cat([original_image_emb,
                                        season_prompt_embed,
                                        weather_prompt_embed,
                                        time_prompt_embed], dim=1)

                prompt_emb.to(train_device)

                pred_ginit, pred_gportion = model(prompt_emb)

                loss, _g, _p, _ccs, _dcs = criterion(image_dirs=image_dirs,
                                                     real_images=real_images,
                                                     prompts=style_prompts, 
                                                     g_init=pred_ginit,
                                                     g_portion= pred_gportion)
                t.set_postfix(loss=loss.item())

                #edited_imgs = [T.ToPILImage()(latent) for latent in _g]
                #breakpoint()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                predicts = pred_ginit*pred_gportion.squeeze()
                preds = predicts.detach().cpu().numpy()
                season_ccs = _ccs[0].detach().cpu().numpy()
                weather_ccs = _ccs[1].detach().cpu().numpy()
                time_ccs = _ccs[2].detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()
                
                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init'+ season_prompts[ps]] = preds.item((ps,0))
                    log_conditions_values['g_init'+ weather_prompts[ps]] = preds.item((ps,1))
                    log_conditions_values['g_init'+ time_prompts[ps]] = preds.item((ps,2))
                    g_init = preds.item((ps,0))+preds.item((ps,1))+preds.item((ps,2))
                    log_conditions_values['sum_g_init'] = g_init
                    log_conditions_values['clip cosin similarity'+ season_prompts[ps]] = season_ccs.item(ps)
                    log_conditions_values['clip cosin similarity'+ weather_prompts[ps]] = weather_ccs.item(ps)
                    log_conditions_values['clip cosin similarity'+ time_prompts[ps]] = time_ccs.item(ps)
                    log_conditions_values['dino cosin similarity'] = struc_dcs.item(ps)
                    wandb.log(log_conditions_values)
                    
                wandb.log({"step loss" : loss})
                total_loss += loss.item()
            epoch_loss = total_loss/len(dataloader)
            wandb.log(
                {   "epoch":epoch+1,
                    "loss": epoch_loss,
                }
            )
        
        optimizer_scheduler.step()
        if epoch%2==0:
            valid_epoch_loss = eval(model= model,
                                    criterion=criterion,
                                    data_root='image_data/eval',
                                    conditions=conditions,
                                    save_image_path=f'Evalutate_images_results/{timestamp}',
                                    epoch=epoch,
                                    device=train_device)
            wandb.log(
                    {   "epoch":epoch+1,
                        "valid loss": valid_epoch_loss,
                    }
                )

            if min_val_loss > valid_epoch_loss:
                min_val_loss = valid_epoch_loss
                torch.save(model.state_dict(), f"./ckpts/{timestamp}_model.pt")



if __name__ == '__main__':
    train(Path('image_data/train'),'cuda')
            
            