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
    data_length = config['data_length']
    
    train_data_root = Path(config['train_data_root'])
    train_latent_data = config['train_latent_data']
    train_embedding_data = config['train_embedding_data']

    eval_data_root = config['eval_data_root']
    eval_latent_data= config['eval_latent_data']
    eval_embedding_data = config['eval_embedding_data']
    
    origin_alpha = config['origin_alpha']
    lr = config['learning_rate']
    model_class = config['attention_model']
    zero_init_model = config['model_zero_init']
    
    model_config = config['model']
    init_g = model_config['init_g']
    divide_out = model_config['divide_out']
    num_guidance_info = model_config['num_guidance_info']
    hidden_dim = model_config['hidden_dim']
    num_layers = model_config['num_layers']
    heads = model_config['heads']
    
    loss_config = config['loss']
    pnp_rate = loss_config['pnp_injection_rate']
    lambda_t = loss_config['lambda_text']
    lambda_s = loss_config['lambda_structure']
    dino_thres = loss_config['dino_threshold']
    guid_sche = loss_config['gradient']
    clip_ds_use = loss_config['clip_ds_use']
    negative_clip_use = loss_config['negative_clip_use']
    
    model_name = 'Attention' if model_class else 'Linear'
    model_class = 'zero_init' if zero_init_model else 'half init'
    struct_text = train_embedding_data.split('/')[-1].split('_')[0]
    clip_loss = "clip_ds" if clip_ds_use else "clip"
    timestamp = get_timestamp()
    
    name = f'''work-{timestamp}-{model_name} {num_layers}, in size {hidden_dim}, {model_class} pnp {pnp_rate} alpha {origin_alpha}
               lambda_t {lambda_t}, lambda_s {lambda_s}, dino thres {dino_thres}, 
               t_loss {clip_loss}, init {init_g}, div {divide_out} lr {lr}, s_text {struct_text}, 
               negative_clip {negative_clip_use}, guidance_schedule {guid_sche}, data len {config['data_length']}'''
    
    
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
    #wandb.log(config)
    seed_everything(seed)
    
    domains = [' on a summer day',
                ' on a spring day',
                ' on a winter day',
                ' on an autumn day',
                ' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a clear day',
                ' on a cloudy day',
                ' on a windy day',
                ' at night time',
                ' at sunset',
                ' at daytime']
    
    criterion = Loss(device=device,
                     **loss_config).to(device)
    
    conditioned_prompt_embedds = criterion.prompt_embeds(domains)
    
    train_dataset = DomainChangeDataset(data_directory=train_data_root,
                                        latents_path=train_latent_data,
                                        embedding_path=train_embedding_data,
                                        data_length=data_length,
                                        conditions=domains,
                                        conditions_embedding=conditioned_prompt_embedds)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    eval_dataset = DomainChangeDataset(data_directory=eval_data_root,
                                        latents_path=eval_latent_data,
                                        embedding_path=eval_embedding_data,
                                        data_length=100,
                                        conditions=domains,
                                        conditions_embedding=conditioned_prompt_embedds)
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)
    
    
    # todo: eval data set
    if model_name:
        guidancemodel =  AttentionModel
    else:
        guidancemodel =  GuidanceModel
        model_config['hidden_dim'] = model_config['hidden_dim']*3
        
    model = guidancemodel(**model_config).to(device)

    

    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                            lr_lambda=lambda epoch: config['lr_lambda']*epoch)
    min_val_loss = 1000
    min_loss = 1000
    
    save_root = Path(f'Train_images_results/{timestamp}')
    save_root.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(100):
        print(f"work-{timestamp}, epoch : {epoch}")
        total_loss = 0
        save_dir = save_root/f'epoch-{epoch}'
        save_dir.mkdir(exist_ok=True)
        with tqdm(train_dataloader) as t:
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
                selected_conditions = [domains[i] for i in condition_number]
                real_image_tensor=real_image_tensor.to(device)
                image_embedding=image_embedding.to(device)
                model_input_embedding = model_input_embedding.to(device)
                latents = latents.to(device)
                sd_text_embedding = sd_text_embedding.to(device)
                from_clip_embedding=from_clip_embedding.to(device)
                to_clip_embedding = to_clip_embedding.to(device)


                if model_name:
                    model_input = torch.cat([image_embedding,
                                             #from_clip_embedding,
                                             to_clip_embedding], dim=1).view(len(idx), 2, -1)
                else:
                    model_input = torch.cat([image_embedding,
                                             from_clip_embedding,
                                             to_clip_embedding], dim=1)



                pred_ginit = model(model_input)

                loss, _g, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
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


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds = pred_ginit.squeeze().detach().cpu().numpy()
                ccs = _ccs.detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()

                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init'+ selected_conditions[ps]] = preds.item(ps)
                    log_conditions_values['clip cosin similarity'+ selected_conditions[ps]] = ccs.item(ps)
                    log_conditions_values['dino cosin similarity'] = struc_dcs.item(ps)
                    wandb.log(log_conditions_values)

                if i%10==0:
                    edited_imgs = [T.ToPILImage()(latent) for latent in _g]
                    for a in range(len(edited_imgs)):
                        str_s_ccs = f'{ccs.item(a):.2f}'.replace('.','_')
                        str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                        s = save_dir/f'{idx.item(a):04}-{selected_conditions[a]}-{int(preds.item(a))}-{str_s_ccs}-{str_dcs}.png'
                        edited_imgs[a].save(s)
                        
                wandb.log({"step loss" : loss})
                total_loss += loss.item()
            epoch_loss = total_loss/len(train_dataloader)
            wandb.log(
                {   "epoch":epoch+1,
                    "loss": epoch_loss,
                }
            )
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), f"./ckpts/train_save/{timestamp}_train_model.pt")
        
        optimizer_scheduler.step()
        valid_epoch_loss = linear_eval(model= model,
                                        criterion=criterion,
                                        eval_dataloader=eval_dataloader,
                                        conditions=domains,
                                        epoch=epoch,
                                        save_image_path=f'Evalutate_images_results/{timestamp}',
                                        device=device,
                                        origin_alpha=origin_alpha,
                                        model_class=model_class
                                        )
        wandb.log(
                {   "epoch":epoch+1,
                    "valid loss": valid_epoch_loss,
                }
            )

        if min_val_loss > valid_epoch_loss:
            min_val_loss = valid_epoch_loss
            torch.save(model.state_dict(), f"./ckpts/{timestamp}_{model_name}_model.pt")


if __name__ == '__main__':
    train('configs/linear_config.yaml')
            
            