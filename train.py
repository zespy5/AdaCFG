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
import torchvision.transforms as T
from eval import eval
from PIL import Image
from random import randint



def train(data_root, train_device, eval_device):
    timestamp = get_timestamp()
    
    name = f"work-{timestamp}-condition 3, lambda_t 1, lambda_s 1, dino thres 0.2, init 50, lr 0.00005"
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
    
    seed_everything(23)
    
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
    
    dataset = DomainChangeDataset(data_directory=data_root,
                                  #conditions=conditions,
                                  #device=device,
                                  )

    #condition_prompt_set = dataset.condition_prompt_set
    #prompt_set = dataset.prompt_set
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GuidanceModel(init_g= 50.0,
                          num_guidance_info=3,
                          linear_in_size=3072,
                          num_mlp_layers=4,
                          hidden_act='gelu',
                          device=train_device).to(train_device)
    
    criterion = Loss(lambda_text=1.0,
                     lambda_structure=1.0,
                     device=train_device,
                     data_root=data_root,
                     dino_threshold=0.2,
                     num_condition=3).to(train_device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    lr = 0.00005

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
                ##original_prompts = [generate_prompt(img) for img in real_images]
                
                #select random condition
                ##condition_idxs = torch.randint(num_conditions,(len(idxs),))
                ##np_condition_idxs = condition_idxs.numpy()
                ##style_prompts = [conditions[np_condition_idxs[i]] for i in range(len(idxs))]
                
                season_prompts = [seasons[randint(0,4)] for _ in range(batch_size)]
                weather_prompts = [weathers[randint(0,5)] for _ in range(batch_size)]
                time_prompts = [times[randint(0,4)] for _ in range(batch_size)]
                
                season_prompt_embed = conditioned_prompt_embedds(season_prompts)
                weather_prompt_embed = conditioned_prompt_embedds(weather_prompts)
                time_prompt_embed = conditioned_prompt_embedds(time_prompts)
                
                style_prompts = [season_prompts, weather_prompts, time_prompts]
                
                #make conditioned prompt
                ##construct_prompts = [p.replace(' at night','') for p in original_prompts]
                ##conditioned_prompts = [construct_prompts[i]+style_prompts[i] for i in range(len(idxs))]
                
                #text clip embedding : model inputs
                original_image_emb = original_image_embedds(real_images)
                ##conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
                prompt_emb = torch.cat([original_image_emb,
                                        season_prompt_embed,
                                        weather_prompt_embed,
                                        time_prompt_embed], dim=1)
                prompt_emb.to(train_device)

                predicts = model(prompt_emb)


                loss, _g, _p = criterion(image_dirs=image_dirs,
                                       real_images=real_images,
                                       prompts=style_prompts, 
                                       guidance_info= predicts)
                t.set_postfix(loss=loss.item())

                
                #edited_imgs = [T.ToPILImage()(latent) for latent in _g]
                #breakpoint()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds = predicts.detach().cpu().numpy()
                log_conditions_values ={}
                for ps in range(len(preds)):
                    log_conditions_values['g_init'+ season_prompts[ps]] = preds.item((ps,0))
                    log_conditions_values['g_init'+ weather_prompts[ps]] = preds.item((ps,1))
                    log_conditions_values['g_init'+ time_prompts[ps]] = preds.item((ps,2))

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
            valid_epoch_loss = eval(model= model,
                        data_root='image_data/eval',
                        conditions=conditions,
                        save_image_path=f'Evalutate_images_results/{timestamp}',
                        epoch=epoch,
                        device=eval_device,
                        model_device=train_device)
            wandb.log(
                    {   "epoch":epoch+1,
                        "valid loss": valid_epoch_loss,
                    }
                )

            if min_val_loss > valid_epoch_loss:
                min_val_loss = valid_epoch_loss
                torch.save(model.state_dict(), f"./ckpts/{timestamp}_model.pt")



if __name__ == '__main__':
    train(Path('image_data/train'),'cuda:0', 'cuda:1')
            
            