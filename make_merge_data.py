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
def make_mean_condition(json_dir, conditions, loss):
    large_clip_text_embedding = loss.prompt_embeds
    large_conditions = get_json(json_dir)
    
    mean_embedds = {}
    for con in conditions:
        conditioned_embeddings = large_clip_text_embedding(large_conditions[con])
        mean_embedding = torch.mean(conditioned_embeddings, dim=0)
        mean_embedds[con] = mean_embedding
    
    return mean_embedds
def main():    
    
    loss = Loss(device='cuda',
                   blip_use=True)
    pipe = loss.pipeline
    
    blip_generate_prompt = generate_prompt
    sd_clip_embedding = pipe.get_text_embeds
    large_clip_image_embedding = loss.image_clip_embeds
    large_clip_text_embedding = loss.prompt_embeds
    
    origin_prompt = '' #'a photograph of a street'
    
    conditions = [' on a summer day',
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
    
    conditioned_prompt = [origin_prompt + con for con in conditions]
    conditioned_embeddings = large_clip_text_embedding(conditioned_prompt)
    mean_embeddings = make_mean_condition('configs/large_conditions.json', conditions, loss)
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

            from_prompt = blip_generate_prompt([img])[0]
            image_data['blip_prompt'] = from_prompt
            origin_text_project_embedding = large_clip_text_embedding(from_prompt)
            origin_sd_text_embedding = sd_clip_embedding(from_prompt)
            text_project_embeddings = {'origin' : origin_text_project_embedding}
            sd_text_embeddings = {'origin': origin_sd_text_embedding}

            for c in range(len(conditions)):
                con = conditions[c]
                condition_prompt = conditioned_prompt[c]
                
                #condition_text_project_embedding = large_clip_text_embedding(condition_prompt)
                condition_text_project_embedding = mean_embeddings[con]
                text_project_embeddings[con] = condition_text_project_embedding
                
                condition_sd_text_embedding = sd_clip_embedding(condition_prompt)
                sd_text_embeddings[con] = condition_sd_text_embedding

            image_data['text_project_embeddings'] = text_project_embeddings
            image_data['sd_text_embeddings'] = sd_text_embeddings

            
            config[file_name] = image_data

        return config

    save_root = Path('merged_latents_forwards')
    train_save = save_root/'blip_train_embeddings.pt'
    eval_save = save_root/'blip_eval_embeddings.pt'
    
    train_data = make_config(train_images)
    eval_data = make_config(eval_images)
    torch.save(train_data, train_save)
    torch.save(eval_data, eval_save)
        



if __name__ == "__main__": 
    main()
                
 
                
    
        
    
