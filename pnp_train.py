import dotenv
import os
import wandb
import torch
import numpy as np
from data.Dataset import DomainChangeZeroShotDataset
from util.loss import VVLoss
from util.utils import *
from models.model import *
import yaml
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as T
from util.eval import eval
import argparse

def train(args):
    config_path = args.config
    config = get_config(config_path)
    model_config = config['model']
    loss_config = config['loss']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    device = config['device']
    epochs = config['epoch']
    
    struct_text = config['train_embedding_data'].split('/')[-1].split('_')[0]
    timestamp = get_timestamp()
    
    name = f'''work-{timestamp}-{struct_text}
               negative_clip {loss_config['negative_clip_use']},
               guidance_schedule : {loss_config['gradient']},
               num layer : {model_config['num_layers']}, 
               num_guidacne_info : {model_config['num_guidance_info']},
               in size : {model_config['hidden_dim']},
               pnp : {loss_config['pnp_injection_rate']},
               lambda_text : {loss_config['lambda_text']}, 
               lambda_structure : {loss_config['lambda_structure']},
               lambda_mean : {loss_config['lambda_mean']},
               lambda_negative : {loss_config['lambda_negative']}, 
               init : {model_config['init_g']}, 
               div : {model_config['divide_out']}, 
               lr : {lr}, 
               lr_lambda : {config['lr_lambda']},
               batch_size {batch_size},
               image size {loss_config['image_size']},
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
     
    criterion = VVLoss(device=device,
                     **loss_config).to(device)
    
    train_dataset = DomainChangeZeroShotDataset(data_directory=config['train_data_root'],
                                             latents_path=config['train_latent_data'],
                                             embedding_path=config['train_embedding_data'],
                                             data_length=config['data_length'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_prompts = train_dataset.text_embeddings
    eval_dataset = DomainChangeZeroShotDataset(data_directory=config['eval_data_root'],
                                             latents_path=config['eval_latent_data'],
                                             embedding_path=config['eval_embedding_data'],
                                             data_length=config['valid_data_length'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=10)
    eval_prompts = eval_dataset.text_embeddings
    
    domains = train_dataset.conditions
    

    model_config['hidden_dim'] = model_config['hidden_dim']*model_config['length']
    model = GuidanceModel(**model_config).to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                            lr_lambda=lambda epoch: config['lr_lambda']*epoch)
    min_val_loss = 1000
    min_loss = 1000
    
    save_root = Path(f'Train_images_results/{timestamp}')
    save_root.mkdir(exist_ok=True, parents=True)
    model_save_path = Path('ckpts')
    model_save_path.mkdir(exist_ok=True)
    

    for epoch in range(epochs):
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
                 to_clip_embedding)= inputs
                
                idx = idx.numpy()
                prompt_idx = prompt_idx.numpy()
                selected_conditions = [domains[i] for i in condition_number]
                
                selected_prompts = []
                for j, con in enumerate(selected_conditions):
                    condition_prompts = list(train_prompts[con]['prompt_emb_pair'].keys())
                    selected_prompts.append(condition_prompts[prompt_idx[j]])
                    
                real_image_tensor=real_image_tensor.to(device)
                image_embedding=image_embedding.to(device)
                condition_mean = condition_mean.to(device)
                latents = latents.to(device)
                sd_text_embedding = sd_text_embedding.to(device)
                to_clip_embedding = to_clip_embedding.to(device)
                

                model_input = torch.cat([image_embedding, to_clip_embedding], dim=1)

                pred_ginit, pred_velocity = model(model_input)

                
                loss, _g, _ccs, _dcs = criterion(real_image_tensor=real_image_tensor,
                                                 condition_mean=condition_mean,
                                                 image_latents=latents,
                                                 sd_prompt_embedding=sd_text_embedding,
                                                 to_clip_embedding=to_clip_embedding,
                                                 g_init=pred_ginit,
                                                 velocity=pred_velocity)
                
                t.set_postfix(loss=loss.item())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                preds = pred_ginit.squeeze().detach().cpu().numpy()
                preds_v = pred_velocity.squeeze().detach().cpu().numpy()
                condition_ccs = _ccs.detach().cpu().numpy()
                struc_dcs = _dcs.detach().cpu().numpy()
                
                for ps in range(len(preds)):
                    log_conditions_values ={}
                    log_conditions_values['g_init '+ selected_conditions[ps]] = preds.item(ps)
                    log_conditions_values['velocity '+ selected_conditions[ps]] = preds_v.item(ps)
                    log_conditions_values['clip cosin similarity '+ selected_conditions[ps]] = condition_ccs.item(ps)
                    log_conditions_values['dino cosin similarity'] = struc_dcs.item(ps)
                    wandb.log(log_conditions_values)
                    
                if i%10==0:
                    edited_imgs = [T.ToPILImage()(latent) for latent in _g]
                    for a in range(len(edited_imgs)):
                        str_w_ccs = f'{condition_ccs.item(a):.2f}'.replace('.','_')
                        str_dcs = f'{struc_dcs.item(a):.2f}'.replace('.','_')
                        s = save_dir/f'{idx.item(a):04}-{int(preds.item(a))}-{selected_prompts[a]}-{selected_conditions[a]}-{str_w_ccs}-{str_dcs}.png'
                        edited_imgs[a].save(s)
                    
                wandb.log({"step loss" : loss})
                total_loss += loss.item()
            epoch_loss = total_loss/len(train_dataloader)
            wandb.log(
                {   "epoch":epoch,
                    "loss": epoch_loss,
                }
            )
            if min_loss > epoch_loss:
                train_model_save_path = model_save_path/'train_save'
                train_model_save_path.mkdir(exist_ok=True)
                min_loss = epoch_loss
                train_model_name = train_model_save_path /f'{timestamp}_train_model.pt'
                torch.save(model.state_dict(), train_model_name)
        
        optimizer_scheduler.step()

        valid_epoch_loss = eval(model= model,
                                criterion=criterion,
                                eval_dataloader=eval_dataloader,
                                eval_prompts=eval_prompts,
                                domains=domains,
                                save_image_path=f'Evalutate_images_results/{timestamp}',
                                epoch=epoch,
                                device=device)
        wandb.log(
                {   "epoch":epoch,
                    "valid loss": valid_epoch_loss,
                }
            )

        sche = loss_config['gradient']
        valid_model_save_path = model_save_path/f'{timestamp}-{sche}'
        valid_model_save_path.mkdir(exist_ok=True, parents=True)
        if min_val_loss > valid_epoch_loss:
            min_val_loss = valid_epoch_loss
            valid_model_name = valid_model_save_path/f'{epoch}_model.pt'
            torch.save(model.state_dict(), valid_model_name)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )

    args= parser.parse_args()
    train(args)


            
            