from pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from util.guidance_scheduler import GuidanceScheduler
import torch
from util.loss import Loss
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from util.utils import get_json
from transformers import BlipProcessor, BlipForConditionalGeneration
image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") 
i2t_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to('cuda')
            
@torch.no_grad()
def generate_prompt(image_dirs):
    if isinstance(image_dirs[0], Path):
        images = [Image.open(img) for img in image_dirs]
    else:
        images = image_dirs
    
    text = ["a photography of"]*len(images)
    inputs = image_processor(images, text, 
                                    return_tensors="pt").to('cuda')

    outputs = i2t_model.generate(**inputs)

    captions = [image_processor.decode(out, skip_special_tokens=True) for out in outputs]

    return captions



def main(cat):    
    

    loss = Loss(device='cuda',
                   blip_use=True)
    pipe = loss.pipeline
    
    blip_generate_prompt = generate_prompt
    sd_clip_embedding = pipe.get_text_embeds
    large_clip_image_embedding = loss.image_clip_embeds
    large_clip_text_embedding = loss.prompt_embeds
    
    
    large_conditions = get_json(f'configs/male_prompts.json')

    for key in large_conditions.values():
        v = set(key)
        print(len(v))
        
    conditions = list(large_conditions.keys())

    image_dir = Path(f'unpaired_image_data/{cat}')
    train_dir = image_dir/f'train'
    eval_dir = image_dir/f'valid'

    train_images = sorted([*train_dir.glob('*')])
    eval_images  = sorted([*eval_dir.glob('*')])
     
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
            
            image = Image.open(img)
            image_project_embedding = large_clip_image_embedding(image)
            each_emb['image_project_embedding'] = image_project_embedding

            from_prompt = blip_generate_prompt([img])[0]
            each_emb['blip_prompt'] = from_prompt
            origin_text_project_embedding = large_clip_text_embedding(from_prompt)
            origin_sd_text_embedding = sd_clip_embedding(from_prompt)
            each_emb['clip'] = origin_text_project_embedding
            each_emb['sd_clip'] = origin_sd_text_embedding
            
            image_embedds[file_name] = each_emb
            
        config['image'] = image_embedds

        return config

    cat_name = cat.replace('_','-')
    save_root = Path('merged_latents_forwards')
    train_save = save_root/f'{cat_name}_train_embeddings.pt'
    eval_save = save_root/f'{cat_name}_valid_embeddings.pt'
    
    train_data = make_config(train_images)
    eval_data = make_config(eval_images)
    torch.save(train_data, train_save)
    torch.save(eval_data, eval_save)
        



if __name__ == "__main__": 
    #category = ['summer_winter','day_night','cat_dog', 'horse_zebra']
    #category = ['day_night']
    category = ['male']
    for cat in category:
        main(cat)
                
 
                
    
        
    
