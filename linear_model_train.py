import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from utils.loss import Loss
from utils.utils import *
from models.model import GuidanceModel, AttentionModel
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from eval import linear_eval
from PIL import Image
from random import randint



def train(data_root, train_device):
    timestamp = get_timestamp()
    
    name = f"work-{timestamp}-linear pnp 0.7 lambda_t 3, lambda_s 1, dino thres 0.4, init 200, 0.2 lr 0.0001"
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
    
    domains = [' on a summer day',
               ' on a spring day',
               ' on a winter day',
               ' on a autumn day',
               ' on a rainy day',
               ' on a foggy day',
               ' on a snowy day',
               ' on a sunny day',
               ' on a cloudy day',
               ' at night',
               ' at sunset',
               ' at sunrise',
               ' at daytime']
    
    
    batch_size = 5
    
    dataset = DomainChangeDataset(data_directory=data_root)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GuidanceModel(init_g= 200.0,
                          divide_out=0.2,
                          num_guidance_info=1,
                          linear_in_size=1536,
                          num_mlp_layers=3,
                          hidden_act='gelu',
                          device=train_device).to(train_device)

    
    criterion = Loss(lambda_text=3.0,
                     lambda_structure=1.0,
                     device=train_device,
                     data_root=data_root,
                     dino_threshold=0.4,
                     num_condition=1,
                     generate_condition_prompt=True,
                     pnp_injection_rate=0.7).to(train_device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
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
                
                data_len = len(image_dirs)
                domain_prompts = [domains[randint(0,12)] for _ in range(data_len)]

                domain_prompt_embed = conditioned_prompt_embedds(domain_prompts)
                domain_prompts = [domain_prompts]

                #text clip embedding : model inputs
                original_image_emb = original_image_embedds(real_images)
                ##conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
                prompt_emb = torch.cat([original_image_emb,
                                        domain_prompt_embed], dim=1)

                prompt_emb.to(train_device)

                pred_ginit = model(prompt_emb)

                loss, _g, _p, _ccs, _dcs = criterion(image_dirs=image_dirs,
                                                     real_images=real_images,
                                                     prompts=domain_prompts, 
                                                     g_init=pred_ginit)
                t.set_postfix(loss=loss.item())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds = pred_ginit.squeeze().detach().cpu().numpy()
                ccs = _ccs[0].detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()

                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init'+ domain_prompts[0][ps]] = preds.item(ps)
                    log_conditions_values['clip cosin similarity'+ domain_prompts[0][ps]] = ccs.item(ps)
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
        if epoch%2==0 and epoch!=0:
            valid_epoch_loss = linear_eval(model= model,
                                           criterion=criterion,
                                           data_root='image_data/eval',
                                           conditions=domains,
                                           save_image_path=f'Evalutate_images_results/{timestamp}',
                                           epoch=epoch,
                                           device=train_device,
                                           )
            wandb.log(
                    {   "epoch":epoch+1,
                        "valid loss": valid_epoch_loss,
                    }
                )

            if min_val_loss > valid_epoch_loss:
                min_val_loss = valid_epoch_loss
                torch.save(model.state_dict(), f"./ckpts/{timestamp}_linear_model.pt")



if __name__ == '__main__':
    train(Path('image_data/train'),'cuda')
            
            