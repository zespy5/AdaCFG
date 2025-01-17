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
    
    name = f"work-{timestamp}"
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
    
    dataset = DomainChangeDataset(data_directory=data_root,
                                  conditions=conditions,
                                  device=device)
    real_image_set = dataset.real_image_set
    condition_prompt_set = dataset.condition_prompt_set
    
    
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = GuidanceModel(linear_in_size=768,
                          device=device).to(device)
    
    criterion = Loss(lambda_text=1.0,
                     lambda_structure=5.0,
                     lambda_reg=0.01,
                     device=device,
                     data_root=data_root).to(device)
    
    prompts_embedds = criterion.prompt_embeds
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    with tqdm(dataloader) as t:
        for i, image_idx in enumerate(t):
            
            idxs= image_idx.numpy()
            prompts = [condition_prompt_set[i] for i in idxs]
            prompt_emb = prompts_embedds(prompts)
            prompt_emb.to(device)

            predicts = model(prompt_emb)
            loss, gen_images = criterion(image_idxs=image_idx,
                             prompts=prompts, 
                             guidance_info= predicts)
            t.set_postfix(loss=loss.item())
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = predicts.detach().cpu()
            preds = preds.mean(dim=0).numpy()
            wandb.log(
                {   "step":i+1,
                    "loss": loss,
                    "init guidance": preds[0],
                    "beta start": preds[1],
                    "beta end": preds[2],
                }
            )
    save_root = Path(f'train_save/{timestamp}')
    save_root.mkdir(exist_ok=True, parents=True)
    for a in range(len(gen_images)):
        s = save_root/f'{a:03}.png'
        gen_images[a].save(s)
        
    torch.save(model.state_dict(), f"./ckpts/{timestamp}_model_val_{loss}.pt")
if __name__ == '__main__':
    train('image_data','cuda')
            
            