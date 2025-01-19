import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from utils.loss import Loss
from models.model import GuidanceModel
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path


def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)

def train(data_root, device):
    timestamp = get_timestamp()
    
    name = f"work-{timestamp}-condition input layer4"
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
    
    seed_everything(1)
    
    conditions = [' on a summer day',
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
                 ' at daytime']
    
    log_conditions_values ={'num_inst':0}
    for c in conditions:
        log_conditions_values['g_init'+ c] = 0
        log_conditions_values['b_start'+ c] = 0
        log_conditions_values['b_end'+ c] = 0
    
    dataset = DomainChangeDataset(data_directory=data_root,
                                  conditions=conditions,
                                  device=device)

    #condition_prompt_set = dataset.condition_prompt_set
    prompt_set = dataset.prompt_set
    
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = GuidanceModel(linear_in_size=768,
                          num_mlp_layers=4,
                          device=device).to(device)
    
    criterion = Loss(lambda_text=10.0,
                     lambda_structure=1.0,
                     lambda_reg=0.0,
                     device=device,
                     data_root=data_root).to(device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds(conditions)
    lr = 0.0001

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    num_conditions = len(conditions)
    for epoch in range(50):
        print(f"epoch : {epoch}")
        #dataset.update_condition_set()
        with tqdm(dataloader) as t:
            for i, image_idx in enumerate(t):
                
                idxs= image_idx.numpy()
                prompts = [prompt_set[i] for i in idxs]
                
                condition_idxs = torch.randint(num_conditions,(len(idxs),))
                np_condition_idxs = condition_idxs.numpy()

                conditioned_prompts = [prompts[i]+conditions[np_condition_idxs[i]] for i in range(len(idxs))]

                    
                prompt_emb = conditioned_prompt_embedds[condition_idxs]
                prompt_emb.to(device)

                predicts = model(prompt_emb)
                loss, gen_images = criterion(image_idxs=image_idx,
                                prompts=conditioned_prompts, 
                                guidance_info= predicts)
                t.set_postfix(loss=loss.item())
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds = predicts.detach().cpu().numpy()
                for ps in range(len(preds)):
                    log_conditions_values['num_inst']+1
                    log_conditions_values['g_init'+ conditions[np_condition_idxs[ps]]] = preds[ps][0]
                    log_conditions_values['b_start'+ conditions[np_condition_idxs[ps]]] = preds[ps][1]
                    log_conditions_values['b_end'+ conditions[np_condition_idxs[ps]]] = preds[ps][2]
                    wandb.log(log_conditions_values)

                
                wandb.log({"step loss" : loss})
            wandb.log(
                {   "epoch":epoch+1,
                    "loss": loss,
                }
            )
        save_root = Path(f'train_save/{timestamp}')
        save_root.mkdir(exist_ok=True, parents=True)
        for a in range(len(gen_images)):
            s = save_root/f'{a:03}.png'
            gen_images[a].save(s)
            
        torch.save(model.state_dict(), f"./ckpts/{timestamp}_model_epoch_{epoch}_val_{loss}.pt")
    
if __name__ == '__main__':
    train('image_data','cuda')
            
            