from util.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from models.model import GuidanceModel
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import torch
import re
from tqdm import tqdm
from PIL import Image
import numpy

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

@torch.no_grad()
def main(model, model_number):
    
    
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
    conditions = ['a photograph of a street'+con for con in conditions]
    model_text_input = prompt_embeds(conditions)

    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/f'val-{model_number}'
    save_valid.mkdir(exist_ok=True)
    

    pnp = PnPPipeline()
    guidance_scheduler = GuidanceScheduler(gradient='constant')
    
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])[:25]
    
    
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
        save_origin_image = save_images/'origin.png'
        image = Image.open(img_dir)
        image.save(save_origin_image)
        
        image_embedding = image_clip_embeds(image).repeat(len(conditions),1)
        
        model_input = torch.cat([image_embedding, model_text_input], dim=1)
        guidance_value = model(model_input)
        guidance = guidance_scheduler.get_guidance_scales(guidance_value)
        
        
        images = [img_dir]*len(conditions)
        outputs = pnp(image_dirs=images,
                      negative_prompt='ugly, blurry, low res, unrealistic, paint',
                      #blip_conditions=[conditions],
                      prompts=[conditions],
                      guidance_scales=guidance,
                      latents_save_root='eval_latents_forward')
        
        guidance_values = guidance_value.squeeze().cpu().numpy()
        gen_images = outputs.images
        prompts = outputs.prompts[0]

        for i in range(len(conditions)):
            g = f'{guidance_values.item(i):.2f}'.replace('.','_')
            save_gen_img = save_images/f'guid-{g}-prompt-{prompts[i]}.png'
            gen_images[i].save(save_gen_img)
            
@torch.no_grad()
def main2(model, model_number):
    
    
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
    conditions = ['a photograph of a street'+con for con in conditions]
    weather = conditions[:9]
    time = conditions[9:]
    

    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)
    
    save_valid = save_root/f'multi-val-{model_number}'
    save_valid.mkdir(exist_ok=True)
    
    
    pnp = PnPPipeline()
    guidance_scheduler = GuidanceScheduler(gradient='constant')
    
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])[:25]
    
    
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
        for t in time:
            save_time = save_images/f'{t}'
            save_time.mkdir(exist_ok=True)
            pnp_multi_cons = [[t]*len(weather), weather]
            con_for_emb = [t]*len(weather)+ weather
            model_text_input = prompt_embeds(con_for_emb)  
              
            save_origin_image = save_time/'origin.png'
            image = Image.open(img_dir)
            image.save(save_origin_image)
            
            image_embedding = image_clip_embeds(image).repeat(len(con_for_emb),1)
            
            model_input = torch.cat([image_embedding, model_text_input], dim=1)
            guidance_value = model(model_input)
            time_g, weather_g = guidance_value.chunk(2)
            total_g = time_g+weather_g
            time_portion = time_g/total_g
            weather_portion = 1-time_portion
            
            portions = torch.cat([time_portion, weather_portion], dim=1)
            guidance = guidance_scheduler.get_guidance_scales(total_g)

            images = [img_dir]*len(weather)
            outputs = pnp(image_dirs=images,
                        negative_prompt='ugly, blurry, low res, unrealistic, paint',
                        #blip_conditions=pnp_multi_cons,
                        prompts = pnp_multi_cons,
                        num_condition=2,
                        guidance_scales=guidance,
                        guidance_portion=portions,
                        latents_save_root='eval_latents_forward')
            
            t_values = time_g.squeeze().cpu().numpy()
            w_values = weather_g.squeeze().cpu().numpy()
            gen_images = outputs.images
            time_prompts = outputs.prompts[0]

            for i,w in enumerate(weather):
                t_g = f'{t_values.item(i):.2f}'.replace('.','_')
                w_g = f'{w_values.item(i):.2f}'.replace('.','_')
                save_gen_img = save_time/f'guid-{t_g}-{w_g}-prompt-{time_prompts[i]}-{w}.png'
                gen_images[i].save(save_gen_img)
        
        
        
    
if __name__ == '__main__':
    model_path = Path('ckpts/0222154441_linear_model.pt')

    time = model_path.stem.split('_')[0]
    model = GuidanceModel(init_g=50.0,
                          divide_out=0.2,
                          hidden_dim=768*2,
                          num_layers=5).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    main2(model,time)
    main(model, time)