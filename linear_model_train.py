import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeDataset
from util.loss import Loss
from util.utils import *
from models.model import *
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from eval import linear_eval
from PIL import Image
from random import randint



def train(config_path):
    config = get_config(config_path)
    
    seed = config['seed']
    device = config['device']
    batch_size = config['batch_size']
    train_data_root = Path(config['train_data_root'])
    eval_data_root = config['eval_data_root']
    origin_alpha = config['origin_alpha']
    lr = config['learning_rate']
    nu_init_text = config['nu_init_text']
    guid_model = config['model_zero_init']
    
    model_config = config['model']
    init_g = model_config['init_g']
    divide_out = model_config['divide_out']
    
    loss_config = config['loss']
    pnp_rate = loss_config['pnp_injection_rate']
    lambda_t = loss_config['lambda_text']
    lambda_s = loss_config['lambda_structure']
    dino_thres = loss_config['dino_threshold']
    dino_loss_use = loss_config['dino_loss_use']
    blip_use = loss_config['blip_use']
    guid_sche_use = loss_config['guidance_schedule_use']
    
    
    model_class = 'zero_init' if guid_model else 'half init'
    struct_text = 'blip ' if blip_use else nu_init_text
    structure_loss = "dino CosinSim" if dino_loss_use else "keys_ssim" 
    
    timestamp = get_timestamp()
    
    name = f'''work-{timestamp}-linear increase {model_class} pnp {pnp_rate} alpha {origin_alpha}
               lambda_t {lambda_t}, lambda_s {lambda_s}, dino thres {dino_thres} sqrt, loss {structure_loss},
               init {init_g}, div {divide_out} lr {lr}, s_text {struct_text}, guidance_schedule_use {guid_sche_use}'''
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
    
    seed_everything(seed)
    
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
    
    
    dataset = DomainChangeDataset(data_directory=train_data_root, data_length=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    guidancemodel = GuidanceModel2 if guid_model else GuidanceModel
    model = guidancemodel(**model_config).to(device)

    criterion = Loss(**loss_config).to(device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds
    original_image_embedds = criterion.image_clip_embeds
    
    if loss_config['blip_use']:
        generate_prompt = criterion.pipeline.generate_prompt
    

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
                image_dirs = [train_data_root/f'{idx:04}.jpg' for idx in idxs]
                real_images = [Image.open(img).convert('RGB') for img in image_dirs]
                
                data_len = len(image_dirs)
                selected_prompts = [domains[randint(0,12)] for _ in range(data_len)]
                
                if blip_use:
                    assert nu_init_text == "", "nu_init_text must be null text"
                    original_prompts = generate_prompt(real_images)
                    blip_domain_prompts = [original_prompts[i]+selected_prompts[i] for i in range(data_len)]
                    domain_prompt_embed = conditioned_prompt_embedds(blip_domain_prompts)
                    domain_prompts = selected_prompts
                elif nu_init_text != "":
                    domain_prompts = [nu_init_text+selected_prompts[i] for i in range(data_len)]
                    domain_prompt_embed = conditioned_prompt_embedds(domain_prompts)
                else:
                    domain_prompts = selected_prompts
                    domain_prompt_embed = conditioned_prompt_embedds(selected_prompts)
                
                input_prompts = [domain_prompts]

                #text clip embedding : model inputs
                original_image_emb = original_image_embedds(real_images)
                ##conditioned_prompt_emb = conditioned_prompt_embedds(conditioned_prompts)
                prompt_emb = torch.cat([original_image_emb,
                                        domain_prompt_embed], dim=1)

                prompt_emb.to(device)

                pred_ginit = model(prompt_emb)

                loss, _g, _p, _ccs, _dcs = criterion(image_dirs=image_dirs,
                                                     real_images=real_images,
                                                     prompts=input_prompts, 
                                                     g_init=pred_ginit,
                                                     origin_alpha=origin_alpha,
                                                     g_portion=None)
                t.set_postfix(loss=loss.item())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds = pred_ginit.squeeze().detach().cpu().numpy()
                ccs = _ccs[0].detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()

                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init'+ selected_prompts[ps]] = preds.item(ps)
                    log_conditions_values['clip cosin similarity'+ selected_prompts[ps]] = ccs.item(ps)
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
            valid_epoch_loss = linear_eval(model= model,
                                           criterion=criterion,
                                           data_root=eval_data_root,
                                           conditions=domains,
                                           save_image_path=f'Evalutate_images_results/{timestamp}',
                                           epoch=epoch,
                                           device=device,
                                           nu_init_text=nu_init_text,
                                           origin_alpha=origin_alpha
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
    train('configs/linear_config.yaml')
            
            