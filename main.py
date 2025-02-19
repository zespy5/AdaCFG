from pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from util.guidance_scheduler import GuidanceScheduler
import torch
from util.loss import Loss
from tqdm import tqdm

def main():    
    loss = Loss(device='cuda',
                   blip_use=True)
    pipe = loss.pipeline

    blip_prompt = pipe.generate_prompt
    sd_clip_embedding = pipe.get_text_embeds
    large_clip_image_embedding = loss.image_clip_embeds
    large_clip_text_embedding = loss.prompt_embeds
    conditions = [' on a summer day',
                ' on a spring day',
                ' on a winter day',
                ' on an autumn day',
                ' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a sunny day',
                ' on a cloudy day',
                ' at night time',
                ' at sunset',
                ' at daytime']

    image_dir = Path('image_data')
    train_dir = image_dir/'train'
    eval_dir = image_dir/'eval'

    train_images = sorted([*train_dir.glob('*')])
    eval_images =sorted([*eval_dir.glob('*')])
    
    def make_config(images):
        config = {}
        for img in tqdm(images):
            file_name = img.stem
            
            
            image_data = {}
            
            
            image = Image.open(img)
            image_project_embedding = large_clip_image_embedding(image)
            image_data['image_project_embedding'] = image_project_embedding 

            #origin_prompt = blip_prompt([image])[0]
            origin_prompt = 'a photograph'
            image_data['origin_prompt'] = origin_prompt
            origin_text_project_embedding = large_clip_text_embedding(origin_prompt)
            origin_sd_text_embedding = sd_clip_embedding(origin_prompt)

            text_project_embeddings = {'origin' : origin_text_project_embedding}
            sd_text_embeddings = {'origin' : origin_sd_text_embedding}

            for con in conditions:
                condition_prompt = origin_prompt + con
                
                condition_text_project_embedding = large_clip_text_embedding(condition_prompt)
                text_project_embeddings[con] = condition_text_project_embedding
                
                condition_sd_text_embedding = sd_clip_embedding(condition_prompt)
                sd_text_embeddings[con] = condition_sd_text_embedding

            image_data['text_project_embeddings'] = text_project_embeddings
            image_data['sd_text_embeddings'] = sd_text_embeddings

            
            config[file_name] = image_data
        
        return config

    save_root = Path('merged_latents_forwards')
    train_save = save_root/'photograph_train_embeddings.pt'
    eval_save = save_root/'photograph_eval_embeddings.pt'
    
    train_data = make_config(train_images)
    eval_data = make_config(eval_images)
    torch.save(train_data, train_save)
    torch.save(eval_data, eval_save)
        



if __name__ == "__main__": 
    main()
                
 
                
    
        
    
