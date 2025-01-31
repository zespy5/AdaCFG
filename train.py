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



def train(data_root, train_device, eval_device):
    timestamp = get_timestamp()
    
    name = f"work-{timestamp}-lambda_t 2.5, lambda_s 1,max 0.2, init 100, lr 0.00005"
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
                                  #conditions=conditions,
                                  #device=device,
                                  )

    #condition_prompt_set = dataset.condition_prompt_set
    #prompt_set = dataset.prompt_set
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = GuidanceModel(linear_in_size=1536,
                          init_g= 100.0,
                          num_mlp_layers=3,
                          device=train_device).to(train_device)
    
    criterion = Loss(lambda_text=2.5,
                     lambda_structure=1.0,
                     device=train_device,
                     data_root=data_root).to(train_device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    lr = 0.00005

    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    min_val_loss = 1000
    
    num_conditions = len(conditions)
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
                original_prompts = [generate_prompt(img) for img in real_images]
                
                #select random condition
                condition_idxs = torch.randint(num_conditions,(len(idxs),))
                np_condition_idxs = condition_idxs.numpy()
                style_prompts = [conditions[np_condition_idxs[i]] for i in range(len(idxs))]
                
                #make conditioned prompt
                construct_prompts = [p.replace(' at night','') for p in original_prompts]
                conditioned_prompts = [construct_prompts[i]+style_prompts[i] for i in range(len(idxs))]
                
                #text clip embedding : model inputs
                original_image_emb = original_image_embedds(real_images)
                conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
                prompt_emb = torch.cat([original_image_emb, conditioned_prompt_emb], dim=1)
                prompt_emb.to(train_device)

                predicts = model(prompt_emb)
                
                

                loss, _g, _p = criterion(image_dirs=image_dirs,
                                       real_images=real_images,
                                       prompts=conditioned_prompts, 
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
                    log_conditions_values['g_init'+ conditions[np_condition_idxs[ps]]] = preds.item((ps,0))

                wandb.log(log_conditions_values)
                wandb.log({"step loss" : loss})
                total_loss += loss.item()
            epoch_loss = total_loss/len(dataloader)
            wandb.log(
                {   "epoch":epoch+1,
                    "loss": epoch_loss,
                }
            )
        
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
            
            