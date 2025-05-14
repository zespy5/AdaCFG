from util.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from models.model import *
from transformers import CLIPModel, CLIPProcessor,  BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image
import numpy
from torch.nn.functional import cosine_similarity
from util.utils import get_json
from ip2p_hack import InstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
import pandas as pd
from util.metric import (Clip, Dino, 
                         Clip_txt_mean, Clip_txt_mean_sim, 
                         prompt_embeds, image_clip_embeds)
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

#@torch.no_grad()
def main(model, grad):   
    condict = {'clear day': 'a photo of a street on a clear day',
                  'cloudy day' : 'a photo of a street on a cloudy day' ,
                  'foggy day' : 'a photo of a street on a foggy day',
                  'rainy day' : 'a photo of a street on a rainy day',
                  'snowy day' : 'a photo of a street on a snowy day',
                  'night':'a photo of a street at night',
                  'sunset':'a photo of a street at sunset'}
    conditions = get_json('configs/ip2p_conditions.json')
    
    save_root = Path('check_valid/ip2p15-3')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/f'candidates'
    save_valid.mkdir(exist_ok=True)
    
    save_pick = save_root/'picked'
    save_pick.mkdir(exist_ok=True)
    

    model_id = "timbrooks/instruct-pix2pix"
    pipe = InstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    guidance_scheduler = GuidanceScheduler(gradient=grad)
    sd_clip_embedding = pipe._encode_prompt
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])
    
    for k in condict.values():
        save_category = save_pick/k
        save_category.mkdir(exist_ok=True)
        
    negative_prompt = f'ugly, blurry, low resolution, unrealistic, paint, distortion, black and white photograph'
    metric_dict={}
    
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
        save_origin_image = save_images/'origin.png'
        image = Image.open(img_dir)
        image.save(save_origin_image)
        image_embedding = image_clip_embeds(image)
        image_tensor = ToTensor()(image)
        
        for k,v in conditions.items():
            v=v[:4]
            save_category = save_images/k
            save_category.mkdir(exist_ok=True)
            batch_size = len(v)
            #ip2p_prompt_embedding = sd_clip_embedding(v, 'cuda', 1, True).chunk(3)[0]
            #breakpoint()
            to_clip_embedding = prompt_embeds(v)
            img_embs = image_embedding.repeat(batch_size,1)
            mean_embedding = to_clip_embedding.mean(dim=0)
            
            model_input = torch.cat([img_embs, to_clip_embedding], dim=1)
            guidance_value = model(model_input)
            guidance = guidance_scheduler.get_guidance_scales(guidance_value)

            images = [image_tensor]*batch_size
            outputs = pipe(#prompt_embeds = ip2p_prompt_embedding,
                           prompt=v,
                           image = images,
                           num_inference_steps=50,
                           guidance_scale=guidance,
                           image_guidance_scale=2,
                           output_type='pt')
                           #negative_prompt=['ugly, blurry, low resolution, unrealistic, paint, distortion']*batch_size)
            guidance_values = guidance_value.squeeze().cpu().numpy()
            gen_images = outputs.images

            losses = []
            for i in range(batch_size):
                m_clip = Clip_txt_mean_sim(mean_embedding, gen_images[i])
                p_clip = Clip(gen_images[i], condict[k])
                n_clip = Clip(gen_images[i], negative_prompt)
                dino = Dino(image, gen_images[i])
                loss = (1-p_clip) + n_clip + (1-dino)*0.1
                g = guidance_values.item(i)
                save_gen_img = save_category/f'prompt-{v[i]}.png'
                gen_images[i].save(save_gen_img)
                
                _metrics = {'guidance': g,
                        'mean clip': m_clip,
                        'positive clip': p_clip,
                        'negative clip': n_clip,
                        'dino': dino}
                losses.append(loss)
                metric_dict[save_gen_img.as_posix()] = _metrics
            min_loss = min(losses)
            min_index = losses.index(min_loss)
            min_image = gen_images[min_index]
            save_picked_image = save_pick/condict[k]/f'{image_name}.png'
            min_image.save(save_picked_image)
            
        df = pd.DataFrame.from_dict(metric_dict, orient='index')
        df.index.name = 'file_name'
        df.to_csv(save_valid/f'metrics.csv')

    
if __name__ == '__main__':
    model_path = Path('ckpts/best_ckpts/15_linear_weather.pt')
    time = model_path.as_posix().split('/')[-2]

    model = GuidanceModel(init_g=25.0,
                          divide_out=0.1,
                          hidden_dim=768*2,
                          num_layers=5,
                          num_guidance_info=1).to('cuda')

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    main(model,'decrease')
