import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainMultiChangeDataset
from util.loss import Loss, CLIPOnlyLoss
from util.utils import *
from models.model import *
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from eval import eval
from PIL import Image
from random import randint



def train(config_path):
    config = get_config(config_path)
    model_config = config['model']
    loss_config = config['loss']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    device = config['device']
    
    
    model_name = 'Attention' if config['attention_model'] else 'Linear'
    model_class = 'zero_init' if config['model_zero_init'] else 'half init'
    struct_text = config['train_embedding_data'].split('/')[-1].split('_')[0]
    structure_loss = "dino CosinSim" if loss_config['dino_loss_use'] else "keys_ssim" 
    clip_loss = "clip_ds" if loss_config['clip_ds_use'] else "clip"
    
    timestamp = get_timestamp()
    
    name = f'''work-{timestamp}-Multi-Condition
               {model_name} {model_config['num_layers']}, in size {model_config['hidden_dim']}, {model_class},
               pnp {loss_config['pnp_injection_rate']} alpha {config['origin_alpha']}, lambda_t {loss_config['lambda_text']}, 
               lambda_s {loss_config['lambda_structure']}, dino thres {loss_config['dino_threshold']}, s_loss {structure_loss}, 
               t_loss {clip_loss}, init {model_config['init_g']}, div {model_config['divide_out']} 
               lr {lr}, s_text {struct_text}, negative_clip {loss_config['negative_clip_use']}, guidance_schedule {loss_config['gradient']},
               data len {config['data_length']}'''
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
    
    seed_everything(config['seed'])
    
    weathers = [' on a summer day',
                ' on a spring day',
                ' on a winter day',
                ' on an autumn day',
                ' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a clear day',
                ' on a cloudy day',]
    
    times = [' at night time',
             ' at sunset',
             ' at daytime']
    
    
    loss_class = Loss if loss_config['dino_loss_use'] else CLIPOnlyLoss
    criterion = loss_class(device=device,
                     **loss_config).to(device)
    
    time_prompt_embedds = criterion.prompt_embeds(times)
    weather_prompt_embedds = criterion.prompt_embeds(weathers)
    
    train_dataset = DomainMultiChangeDataset(data_directory=config['train_data_root'],
                                             latents_path=config['train_latent_data'],
                                             embedding_path=config['train_embedding_data'],
                                             data_length=config['data_length'],
                                             time_conditions=times,
                                             time_conditions_embedding=time_prompt_embedds,
                                             weather_conditions=weathers,
                                             weather_conditions_embedding=weather_prompt_embedds)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    eval_dataset = DomainMultiChangeDataset(data_directory=config['eval_data_root'],
                                             latents_path=config['eval_latent_data'],
                                             embedding_path=config['eval_embedding_data'],
                                             data_length=100,
                                             time_conditions=times,
                                             time_conditions_embedding=time_prompt_embedds,
                                             weather_conditions=weathers,
                                             weather_conditions_embedding=weather_prompt_embedds)
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)

    if loss_config['clip_ds_use']:                 
        model = MultiConditionAttentionModel(**model_config).to(device)
    else:
        model = MultiConditionAttentionModel2(**model_config).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                            lr_lambda=lambda epoch: config['lr_lambda']*epoch)
    min_val_loss = 1000

    for epoch in range(100):
        print(f"work-{timestamp}, epoch : {epoch}")
        total_loss = 0
        with tqdm(train_dataloader) as t:
            for i, inputs in enumerate(t):
                (idx,
                 time_condition_number,
                 weather_condition_number,
                 real_image_tensor,
                 image_embedding,
                 model_time_input_embedding,
                 model_weather_input_embedding,
                 latents,
                 time_sd_text_embedding,
                 weather_sd_text_embedding,
                 from_clip_embedding,
                 to_time_clip_embedding,
                 to_weather_clip_embedding)= inputs
                
                selected_time_conditions = [times[i] for i in time_condition_number]
                selected_weather_conditions = [weathers[i] for i in weather_condition_number]
                real_image_tensor=real_image_tensor.to(device)
                image_embedding=image_embedding.to(device)
                #model_time_input_embedding = model_time_input_embedding.to(device)
                #model_weather_input_embedding = model_weather_input_embedding.to(device)
                latents = latents.to(device)
                time_sd_text_embedding = time_sd_text_embedding.to(device)
                weather_sd_text_embedding = weather_sd_text_embedding.to(device)
                from_clip_embedding=from_clip_embedding.to(device)
                to_time_clip_embedding = to_time_clip_embedding.to(device)
                to_weather_clip_embedding = to_weather_clip_embedding.to(device)
                
                if loss_config['clip_ds_use']:
                    model_input = torch.cat([image_embedding,
                                            from_clip_embedding,
                                            to_time_clip_embedding,
                                            to_weather_clip_embedding], dim=1).view(len(idx), 4, -1)
                else:
                    model_input = torch.cat([image_embedding,
                                             to_time_clip_embedding,
                                             to_weather_clip_embedding], dim=1).view(len(idx), 3, -1)

                pred_ginit, pred_portion = model(model_input)

                to_clip_embedding = torch.cat([to_time_clip_embedding, to_weather_clip_embedding])
                sd_text_embedding = torch.cat([time_sd_text_embedding, weather_sd_text_embedding])
                
                
                loss, _g, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                 clip_real_image_embedding=image_embedding,
                                                 from_clip_embedding=from_clip_embedding,
                                                 to_clip_embedding=to_clip_embedding,
                                                 model_input_embedding=None,
                                                 image_latents=latents,
                                                 sd_prompt_embedding=sd_text_embedding,
                                                 g_init=pred_ginit,
                                                 origin_alpha=config['origin_alpha'],
                                                 g_portion=pred_portion)
                
                t.set_postfix(loss=loss.item())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                time_ccs, weather_ccs = _ccs.chunk(2)
                predicts = pred_ginit*pred_portion.squeeze()
                preds = predicts.detach().cpu().numpy()
                weather_ccs = weather_ccs.detach().cpu().numpy()
                time_ccs = time_ccs.detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()
                
                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init'+ selected_weather_conditions[ps]] = preds.item((ps,1))
                    log_conditions_values['g_init'+ selected_time_conditions[ps]] = preds.item((ps,0))
                    g_init = preds.item((ps,0))+preds.item((ps,1))
                    log_conditions_values['sum_g_init'] = g_init
                    log_conditions_values['clip cosin similarity'+ selected_weather_conditions[ps]] = weather_ccs.item(ps)
                    log_conditions_values['clip cosin similarity'+ selected_time_conditions[ps]] = time_ccs.item(ps)
                    log_conditions_values['dino cosin similarity'] = struc_dcs.item(ps)
                    wandb.log(log_conditions_values)
                    
                wandb.log({"step loss" : loss})
                total_loss += loss.item()
            epoch_loss = total_loss/len(train_dataloader)
            wandb.log(
                {   "epoch":epoch+1,
                    "loss": epoch_loss,
                }
            )
        
        optimizer_scheduler.step()

        valid_epoch_loss = eval(model= model,
                                criterion=criterion,
                                eval_dataloader=eval_dataloader,
                                times= times,
                                weathers=weathers,
                                save_image_path=f'Evalutate_images_results/{timestamp}',
                                epoch=epoch,
                                origin_alpha=config['origin_alpha'],
                                device=device)
        wandb.log(
                {   "epoch":epoch+1,
                    "valid loss": valid_epoch_loss,
                }
            )

        if min_val_loss > valid_epoch_loss:
            min_val_loss = valid_epoch_loss
            torch.save(model.state_dict(), f"./ckpts/{timestamp}_model.pt")
                
    torch.save(model.state_dict(), f"./ckpts/{timestamp}_{model_name}_model_last.pt")


if __name__ == '__main__':
    train('configs/multi_condition_config.yaml')
            
            