import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeBLIPZeroShotDataset
from util.loss import BLIPLoss
from util.utils import *
from models.model import *
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from eval import blip_eval
from PIL import Image
from random import randint



def train(config_path):
    config = get_config(config_path)
    model_config = config['model']
    loss_config = config['loss']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    device = config['device']
    
    struct_text = config['train_embedding_data'].split('/')[-1].split('_')[0]
    clip_loss = "clip_ds" if loss_config['clip_ds_use'] else "clip"
    
    timestamp = get_timestamp()
    
    name = f'''work-{timestamp}-BLIP-Condition
               num layer : {model_config['num_layers']}, 
               in size : {model_config['hidden_dim']},
               heads : {model_config['heads']},
               pnp : {loss_config['pnp_injection_rate']},
               lambda_text : {loss_config['lambda_text']}, 
               lambda_structure : {loss_config['lambda_structure']},
               lambda_blip : {loss_config['lambda_blip']},
               lambda_negative : {loss_config['lambda_negative']},
               dino thres : {loss_config['dino_threshold']},
               t_loss : {clip_loss}, 
               blip init {model_config['init_blip_g']}, 
               init {model_config['init_g']}, 
               div {model_config['divide_out']} 
               lr {lr}, 
               s_text {struct_text}, 
               negative_clip {loss_config['negative_clip_use']}, 
               guidance_schedule {loss_config['gradient']},
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

       
    criterion = BLIPLoss(device=device,
                     **loss_config).to(device)
    
    train_dataset = DomainChangeBLIPZeroShotDataset(data_directory=config['train_data_root'],
                                             latents_path=config['train_latent_data'],
                                             embedding_path=config['train_embedding_data'],
                                             data_length=config['data_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_prompts = train_dataset.text_embeddings
    eval_dataset = DomainChangeBLIPZeroShotDataset(data_directory=config['eval_data_root'],
                                             latents_path=config['eval_latent_data'],
                                             embedding_path=config['eval_embedding_data'],
                                             data_length=100)
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)
    eval_prompts = eval_dataset.text_embeddings
    
    domains = train_dataset.conditions
    
    model = MultiConditionAttentionBLIPModel(**model_config).to(device)



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
                 prompt_idx,
                 condition_number,
                 real_image_tensor,
                 image_embedding,
                 condition_mean,
                 latents,
                 sd_text_embedding,
                 to_clip_embedding,
                 blip_sd_embedding,
                 blip_clip_embedding)= inputs
                
                idx = idx.numpy()
                prompt_idx = prompt_idx.numpy()
                selected_conditions = [domains[i] for i in condition_number]
                
                selected_prompts = []
                for j, con in enumerate(selected_conditions):
                    condition_prompts = list(train_prompts[con]['prompt_emb_pair'].keys())
                    selected_prompts.append(condition_prompts[prompt_idx[j]])
                    
                real_image_tensor=real_image_tensor.to(device)
                image_embedding=image_embedding.to(device)
                #condition_mean = condition_mean.to(device)
                latents = latents.to(device)
                sd_text_embedding = sd_text_embedding.to(device)
                to_clip_embedding = to_clip_embedding.to(device)
                blip_sd_embedding = blip_sd_embedding.to(device)
                blip_clip_embedding = blip_clip_embedding.to(device)
                

                model_input = torch.cat([image_embedding,
                                         blip_clip_embedding,
                                         to_clip_embedding], dim=1).view(len(idx), model_config['length'], -1)

                pred_blip_ginit, pred_ginit = model(model_input)
                total_g = pred_blip_ginit+pred_ginit
                portion1 = pred_blip_ginit/total_g
                portion2 = 1-portion1
                portion = torch.cat([portion1, portion2], dim=1)
                
                sd_text_embedding = torch.cat([blip_sd_embedding, sd_text_embedding])
                
                loss, _g, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                 from_clip_embedding=blip_clip_embedding,
                                                 to_clip_embedding=to_clip_embedding,
                                                 image_latents=latents,
                                                 sd_prompt_embedding=sd_text_embedding,
                                                 g_init=total_g,
                                                 g_portion=portion)
                
                t.set_postfix(loss=loss.item())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                
                blip_ccs, condition_ccs = _ccs.chunk(2)
                predicts = total_g*portion.squeeze()
                preds = predicts.detach().cpu().numpy()
                condition_ccs = condition_ccs.detach().cpu().numpy()
                blip_ccs = blip_ccs.detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()
                
                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init '+ selected_conditions[ps]] = preds.item((ps,1))
                    log_conditions_values['g_init blip'] = preds.item((ps,0))
                    g_init = preds.item((ps,0))+preds.item((ps,1))
                    log_conditions_values['sum_g_init'] = g_init
                    log_conditions_values['clip cosin similarity '+ selected_conditions[ps]] = condition_ccs.item(ps)
                    log_conditions_values['clip cosin similarity blip'] = blip_ccs.item(ps)
                    log_conditions_values['dino cosin similarity'] = struc_dcs.item(ps)
                    wandb.log(log_conditions_values)
                    
                if i%10==0:
                    edited_imgs = [T.ToPILImage()(latent) for latent in _g]
                    for a in range(len(edited_imgs)):
                        str_w_ccs = f'{condition_ccs.item(a):.2f}'.replace('.','_')
                        str_t_ccs = f'{blip_ccs.item(a):.2f}'.replace('.','_')
                        str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                        s = save_dir/f'{idx.item(a):04}-{int(preds.item(a,0))}-{str_t_ccs}-{selected_conditions[a]}-{selected_prompts[a]}-{int(preds.item(a,1))}-{str_w_ccs}-{str_dcs}.png'
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

        valid_epoch_loss = blip_eval(model= model,
                                criterion=criterion,
                                eval_dataloader=eval_dataloader,
                                eval_prompts=eval_prompts,
                                domains=domains,
                                save_image_path=f'Evalutate_images_results/{timestamp}',
                                epoch=epoch,
                                length=model_config['length'],
                                device=device)
        wandb.log(
                {   "epoch":epoch+1,
                    "valid loss": valid_epoch_loss,
                }
            )

        if min_val_loss > valid_epoch_loss:
            min_val_loss = valid_epoch_loss
            torch.save(model.state_dict(), f"./ckpts/{timestamp}_model.pt")
                

if __name__ == '__main__':
    train('configs/blip_multi_config.yaml')
            
            