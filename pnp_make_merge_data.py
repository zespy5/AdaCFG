from util.pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from util.guidance_scheduler import GuidanceScheduler
import torch
from util.loss import VVLoss
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from util.utils import get_json
import argparse

def make_pnp_latents(pnp,
                     image_path : Path, 
                     latents_save_path: Path):
    
    latents_save_path.mkdir(exist_ok=True)
    
    last_step = pnp.inversion_timesteps[-1]
    sorted_images = sorted(list(image_path.glob('*')))
    for img_dir in sorted_images:
        latent_save_dir = latents_save_path/img_dir.stem
        
        source_latent_last_step_file = latent_save_dir/f'noisy_latents_{last_step}.pt'
        if not source_latent_last_step_file.exists():
            latent_save_dir.mkdir(exist_ok=True)
            pnp.extract_latents(img_dir, latent_save_dir)
            
def make_latents_dataset(scheduler, latent_root, save_latent_path):
    latent_paths = sorted([*latent_root.glob('*')])
    latents_dict = {}
    
    for lat in latent_paths:
        image_name = lat.stem
        latents_file_name = [lat/f'noisy_latents_{t}.pt' for t in scheduler.timesteps]
        latents = [torch.load(fn, weights_only=False, map_location='cpu') for fn in latents_file_name]
        e = torch.cat(latents)
        latents_dict[image_name] = e

    torch.save(latents_dict,save_latent_path)

def main(args): 
    prompt_cfg = args.augmented_prompt_path
    image_data_root = args.image_data
    latents_steps = args.latents_steps
    
    data_root = Path(image_data_root)
    save_root = Path('merged_latents_forwards')   
    save_root.mkdir(exist_ok=True)
    
    loss = VVLoss(device='cuda',
                  latents_steps=latents_steps)
    pipe = loss.pipeline
    
    sd_clip_embedding = pipe.get_text_embeds
    large_clip_image_embedding = loss.image_clip_embeds
    large_clip_text_embedding = loss.prompt_embeds
    scheduler = pipe.scheduler
    
    
    large_conditions = get_json(prompt_cfg)
    conditions = list(large_conditions.keys())
    

    image_dir = Path(image_data_root)
    train_dir = image_dir/f'train'
    valid_dir = image_dir/f'valid'
    train_latent_dir = image_dir/f'train_latents'
    valid_latent_dir = image_dir/f'valid_latents'
    
    make_pnp_latents(pipe, valid_dir, valid_latent_dir)
    save_valid_latent_dataset = save_root/f"{data_root.stem}_{valid_latent_dir.stem}.pt"
    make_latents_dataset(scheduler, valid_latent_dir, save_valid_latent_dataset)
    
    
    make_pnp_latents(pipe, train_dir, train_latent_dir)
    save_train_latent_dataset = save_root/f"{data_root.stem}_{train_latent_dir.stem}.pt"
    make_latents_dataset(scheduler, train_latent_dir, save_train_latent_dataset)

    
    train_images = sorted([*train_dir.glob('*')])
    eval_images  = sorted([*valid_dir.glob('*')])
     
    def make_config(images):
        config = {}
        
        text_embedds = {}
        for con in conditions:
            prompts = large_conditions[con]
            conditioned_embeddings = large_clip_text_embedding(prompts)
            mean_embedding = torch.mean(conditioned_embeddings, dim=0)
            
            condition_sd_text_embedding = sd_clip_embedding(prompts)
            
            each_emb = {'mean_embedding': mean_embedding}
            prompt_emb_pair = {}
            for i,prompt in enumerate(prompts):
                
                prompt_emb_pair[prompt] = {'clip':conditioned_embeddings[i],
                                           'sd_clip': condition_sd_text_embedding[i]}
            each_emb['prompt_emb_pair'] = prompt_emb_pair
            text_embedds[con] = each_emb
        
        config['text'] = text_embedds
            
        image_embedds = {}
        for img in tqdm(images):
            file_name = img.stem
            
            each_emb = {}
            
            image = Image.open(img).resize((512,512))
            image_project_embedding = large_clip_image_embedding(image)
            each_emb['image_project_embedding'] = image_project_embedding
            
            image_embedds[file_name] = each_emb
            
        config['image'] = image_embedds

        return config

    train_save = save_root/f'{data_root.stem}_train_embeddings.pt'
    eval_save = save_root/f'{data_root.stem}_valid_embeddings.pt'
    
    train_data = make_config(train_images)
    eval_data = make_config(eval_images)
    torch.save(train_data, train_save)
    torch.save(eval_data, eval_save)
        



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--augmented_prompt_path",
        type=str,
        default="configs/training_conditions.json"
    )
    
    parser.add_argument(
        "--image_data",
        type=str,
        default="image_data"
    )
    
    parser.add_argument(
        "--latents_steps",
        type=int,
        default=50
    )
    
    args= parser.parse_args()
    main(args)
                
 
                
    
        
    
