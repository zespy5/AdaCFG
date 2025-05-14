from pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from util.guidance_scheduler import GuidanceScheduler
import torch
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from util.utils import get_json
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
@torch.no_grad()
def prompt_embeds(prompts):
    clip_inputs = clip_processor(text=prompts,
                                        return_tensors="pt", 
                                        padding=True)
    text_features = clip_model.get_text_features(clip_inputs["input_ids"].to('cuda'),
                                                        clip_inputs["attention_mask"].to('cuda'))
        
    return text_features

@torch.no_grad()
def image_clip_embeds(image):
    clip_inputs = clip_processor(images=image,
                                        return_tensors="pt", 
                                        padding=True)
    image_features = clip_model.get_image_features(clip_inputs["pixel_values"].to('cuda'))
        
    return image_features

def main():    
    
    model_id = "timbrooks/instruct-pix2pix"
    ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None).to("cuda")
    
    
    sd_clip_embedding = ip2p_pipe._encode_prompt
    large_clip_image_embedding = image_clip_embeds
    large_clip_text_embedding = prompt_embeds
    
    
    large_conditions = get_json(f'configs/large_conditions.json')
    instruct_condition = get_json('configs/ip2p_conditions.json')
    for key in large_conditions.values():
        v = set(key)
        print(len(v))
        
    conditions = list(large_conditions.keys())

    image_dir = Path(f'image_data')
    train_dir = image_dir/f'train'
    eval_dir = image_dir/f'eval'

    train_images = sorted([*train_dir.glob('*')])
    eval_images  = sorted([*eval_dir.glob('*')])
     
    def make_config(images):
        config = {}
        
        text_embedds = {}
        for con in conditions:
            prompts = large_conditions[con]
            conditioned_embeddings = large_clip_text_embedding(prompts)
            mean_embedding = torch.mean(conditioned_embeddings, dim=0)
            
            instructs = instruct_condition[con]
            conditioned_embeddings = large_clip_text_embedding(instructs)
            condition_sd_text_embedding = sd_clip_embedding(instructs,'cuda',1,True).chunk(3)[0]
            
            each_emb = {'mean_embedding': mean_embedding}
            prompt_emb_pair = {}
            for i,prompt in enumerate(instructs):
                
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
            
            image_embedds[file_name] = each_emb
            
        config['image'] = image_embedds

        return config

    save_root = Path('merged_latents_forwards')
    train_save = save_root/f'ip2p_train_embeddings.pt'
    eval_save = save_root/f'ip2p_eval_embeddings.pt'
    
    train_data = make_config(train_images)
    torch.save(train_data, train_save)
    del train_data
    eval_data = make_config(eval_images)
    torch.save(eval_data, eval_save)
        



if __name__ == "__main__": 
    main()
                
 
                
    
        
    
