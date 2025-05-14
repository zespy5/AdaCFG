from util.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from models.model import *
from transformers import CLIPModel, CLIPProcessor,  BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import torch
import re
from tqdm import tqdm
from PIL import Image
import numpy
import pandas as pd
from torch.nn.functional import cosine_similarity
from util.utils import get_json
from util.metric import (Clip, Dino, 
                         Clip_txt_mean, Clip_txt_mean_sim, 
                         prompt_embeds, image_clip_embeds)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
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
def main(model, grad, model_number=0):   
    condict = {'clear day':'a photo of a street on a clear day',
                  'cloudy day':'a photo of a street on a cloudy day',
                  'foggy day':'a photo of a street on a foggy day',
                  'rainy day':'a photo of a street on a rainy day',
                  'snowy day':'a photo of a street on a snowy day',
                  'night':'a photo of a street at night',
                  'sunset':'a photo of a street at sunset'}
    conditions = get_json('configs/conditions.json')
    
    save_root = Path('check_valid/sine')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/f'candidates'
    save_valid.mkdir(exist_ok=True)
    
    save_pick = save_root/'picked'
    save_pick.mkdir(exist_ok=True)
    
    for k in condict.values():
        save_category = save_pick/k
        save_category.mkdir(exist_ok=True)

    pnp = PnPPipeline(pnp_attn_t=0.9,
                      pnp_f_t=0.8)
    guidance_scheduler = GuidanceScheduler(gradient=grad, 
                                           lower_bound=3.0)
    
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])
    
    
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
        for k,v in conditions.items():
            
            save_category = save_images/k
            save_category.mkdir(exist_ok=True)
            batch_size = len(v)
            
            to_clip_embedding = prompt_embeds(v)
            img_embs = image_embedding.repeat(batch_size,1)
            mean_embedding = to_clip_embedding.mean(dim=0)
            model_input = torch.cat([img_embs, to_clip_embedding], dim=1)
            guidance_value = model(model_input)
            guidance = guidance_scheduler.get_guidance_scales(guidance_value)
            
            images = [img_dir]*batch_size
            outputs = pnp(image_dirs=images,
                        negative_prompt='ugly, blurry, low resolution, unrealistic, paint, distortion',
                        prompts=[v],
                        guidance_scales=guidance,
                        latents_save_root='eval_latents_forward')
            
            guidance_values = guidance_value.squeeze().cpu().numpy()
            gen_images = outputs.images
            prompts = outputs.prompts[0]

            losses = []
            for i in range(batch_size):
                m_clip = Clip_txt_mean_sim(mean_embedding, gen_images[i])
                p_clip = Clip(gen_images[i], condict[k])
                n_clip = Clip(gen_images[i], negative_prompt)
                dino = Dino(image, gen_images[i])
                loss = (1-p_clip) + n_clip + (1-dino)*0.1
                g = guidance_values.item(i)
                save_gen_img = save_category/f'prompt-{prompts[i]}.png'
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
        
'''@torch.no_grad()
def main(model, grad, model_number=0):   
    condict = {'white_hair': "A photo of a man with white hair",
               'red_hair': "A photo of a man with red dyed hair",
               'blond_hair': "A photo of a man with blond hair",
               'beard': "A photo of a man with a thick beard",
               'dark_skin': "A photo of a man with dark skin",
               'sad_expression': "A photo of a sad man",
               'smiling_expression': "A photo of a man smiling"}
    conditions = get_json('configs/male_conditions.json')
    
    save_root = Path('check_valid/man40')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/f'candidates'
    save_valid.mkdir(exist_ok=True)
    
    save_pick = save_root/'picked'
    save_pick.mkdir(exist_ok=True)
    
    for k in condict.values():
        save_category = save_pick/k
        save_category.mkdir(exist_ok=True)

    pnp = PnPPipeline(pnp_attn_t=0.8,
                      pnp_f_t=0.8)
    guidance_scheduler = GuidanceScheduler(gradient=grad)
    
    data_root = Path('unpaired_image_data/male/test')
    image_dirs = sorted([*data_root.glob('*')])
    
    
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
        for k,v in conditions.items():
            
            save_category = save_images/k
            save_category.mkdir(exist_ok=True)
            batch_size = len(v)
            
            to_clip_embedding = prompt_embeds(v)
            img_embs = image_embedding.repeat(batch_size,1)
            mean_embedding = to_clip_embedding.mean(dim=0)
            model_input = torch.cat([img_embs, to_clip_embedding], dim=1)
            guidance_value = model(model_input)
            guidance = guidance_scheduler.get_guidance_scales(guidance_value)
            
            images = [img_dir]*batch_size
            outputs = pnp(image_dirs=images,
                        negative_prompt='ugly, blurry, low resolution, unrealistic, paint, distortion',
                        prompts=[v],
                        guidance_scales=guidance,
                        latents_save_root='unpaired_image_data/male_latents/test_latents_forward')
            
            guidance_values = guidance_value.squeeze().cpu().numpy()
            gen_images = outputs.images
            prompts = outputs.prompts[0]

            losses = []
            for i in range(batch_size):
                m_clip = Clip_txt_mean_sim(mean_embedding, gen_images[i])
                p_clip = Clip(gen_images[i], condict[k])
                n_clip = Clip(gen_images[i], negative_prompt)
                dino = Dino(image, gen_images[i])
                loss = (1-p_clip) + n_clip + (1-dino)*0.1
                g = guidance_values.item(i)
                save_gen_img = save_category/f'prompt-{prompts[i]}.png'
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
'''

'''@torch.no_grad()
def main(model, model_number,grad):

    
    from_prompt = ''
    conditions = [
                ' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a clear day',
                ' on a cloudy day',
                ' at night time',
                ' at sunset',]

    conditions = [from_prompt+con for con in conditions]
    to_clip_embedding = prompt_embeds(conditions)
    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/f'val-{model_number}-simple'
    save_valid.mkdir(exist_ok=True)


    pnp = PnPPipeline(pnp_attn_t=0.95,
                      pnp_f_t=0.95)
    guidance_scheduler = GuidanceScheduler(gradient=grad)

    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])[:30]


    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)

        save_origin_image = save_images/'origin.png'
        image = Image.open(img_dir)
        image.save(save_origin_image)

        image_embedding = image_clip_embeds(image).repeat(len(conditions),1)

        model_input = torch.cat([image_embedding,
                                to_clip_embedding], dim=1).view(len(conditions), 2, -1)
        guidance_value = model(model_input)
        guidance = guidance_scheduler.get_guidance_scales(guidance_value)


        images = [img_dir]*len(conditions)
        outputs = pnp(image_dirs=images,
                      negative_prompt='ugly, blurry, low resolution, unrealistic, paint, distortion',
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
            '''
@torch.no_grad()
def multi_condition_main(model, model_number,grad):
    
    
    conditions = ['',
                  ' on a summer day',
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
    
    embeddings = prompt_embeds(conditions)
    from_prompt = conditions[0]
    weather = conditions[1:10]
    time = conditions[10:]
    
    from_embedding = embeddings[0].unsqueeze(0)
    weather_embedding = embeddings[1:10]
    time_embedding = embeddings[10:]

    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)
    
    save_valid = save_root/f'multi-val-{model_number}'
    save_valid.mkdir(exist_ok=True)
    
    
    pnp = PnPPipeline()
    guidance_scheduler = GuidanceScheduler(gradient=grad)
    
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])[:25]
    
    
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
        for i,t in enumerate(time):
            
            batch_size = len(weather)
            save_time = save_images/f'{t}'
            save_time.mkdir(exist_ok=True)
            pnp_multi_cons = [[t]*batch_size, weather]
              
            save_origin_image = save_time/'origin.png'
            image = Image.open(img_dir)
            image.save(save_origin_image)
            
            image_embedding=image_clip_embeds(image).repeat(batch_size,1)
            from_emb = from_embedding.repeat(batch_size,1)
            to_time_clip_embedding = time_embedding[i].unsqueeze(0).repeat(batch_size,1)

            model_input = torch.cat([image_embedding,
                                     from_emb,
                                     to_time_clip_embedding,
                                     weather_embedding], dim=1).view(batch_size, 4, -1)

            pred_ginit, pred_portion = model(model_input)
            gs = pred_ginit*pred_portion
            time_g = gs[:,0]
            weather_g = gs[:,1]

            guidance = guidance_scheduler.get_guidance_scales(pred_ginit)

            images = [img_dir]*len(weather)
            outputs = pnp(image_dirs=images,
                        negative_prompt='ugly, blurry, low resolution, unrealistic, paint, distortion',
                        #blip_conditions=pnp_multi_cons,
                        prompts = pnp_multi_cons,
                        num_condition=2,
                        guidance_scales=guidance,
                        guidance_portion=pred_portion,
                        latents_save_root='eval_latents_forward')
            
            t_values = time_g.squeeze().cpu().numpy()
            w_values = weather_g.squeeze().cpu().numpy()
            gen_images = outputs.images
            time_prompts = outputs.prompts[0]
            weather_prompts = outputs.prompts[1]

            for i,w in enumerate(weather):
                t_g = f'{t_values.item(i):.2f}'.replace('.','_')
                w_g = f'{w_values.item(i):.2f}'.replace('.','_')
                save_gen_img = save_time/f'guid-{t_g}-{w_g}-prompt-{time_prompts[i]}-{weather_prompts[i]}.png'
                gen_images[i].save(save_gen_img)
        

           
@torch.no_grad()
def blip_main(model, model_number, grad):
    
    
    from_prompt = 'a photograph of a street'
    conditions = [' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a clear day',
                ' on a cloudy day',
                ' at night time',
                ' at sunset',
                ' at daytime']

    conditions = [from_prompt+con for con in conditions]
    embeddings = prompt_embeds(conditions)
    
    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)
    
    save_valid = save_root/f'blip_val-{model_number}'
    save_valid.mkdir(exist_ok=True)
    
    
    pnp = PnPPipeline(pnp_attn_t=0.95,
                      pnp_f_t=0.95)
    guidance_scheduler = GuidanceScheduler(gradient=grad)
    
    data_root = Path('image_data/eval')
    image_dirs = sorted([*data_root.glob('*')])[:25]
    
    batch_size = len(conditions)
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
    
        save_origin_image = save_images/'origin.png'
        image = Image.open(img_dir)
        image.save(save_origin_image)
        
        image_embedding = image_clip_embeds(image).repeat(batch_size,1)
        
        blip_prompt = generate_prompt([image])
        
        blip_prompt_embedding = prompt_embeds(blip_prompt).repeat(batch_size,1)
        blip_prompts = blip_prompt*batch_size
        pnp_multi_cons = [blip_prompts,conditions]
        
        model_input = torch.cat([image_embedding,
                                    blip_prompt_embedding,
                                    embeddings], dim=1).view(batch_size,3,-1)
        pred_blip_ginit, pred_ginit = model(model_input)
        total_g = pred_blip_ginit+pred_ginit
        portion1 = pred_blip_ginit/total_g
        portion2 = 1-portion1
        portion = torch.cat([portion1, portion2], dim=1)
        
        guidance = guidance_scheduler.get_guidance_scales(total_g)

        images = [img_dir]*batch_size
        outputs = pnp(image_dirs=images,
                    negative_prompt='ugly, blurry, low resolution, unrealistic, paint, distortion',
                    prompts = pnp_multi_cons,
                    num_condition=2,
                    guidance_scales=guidance,
                    guidance_portion=portion,
                    latents_save_root='eval_latents_forward')
        
        t_values = pred_blip_ginit.squeeze().cpu().numpy()
        w_values = pred_ginit.squeeze().cpu().numpy()
        gen_images = outputs.images
        blip_prompts = outputs.prompts[0]
        weather_prompts = outputs.prompts[1]

        for i in range(batch_size):
            t_g = f'{t_values.item(i):.1f}'.replace('.','_')
            w_g = f'{w_values.item(i):.1f}'.replace('.','_')
            save_gen_img = save_images/f'guid-{t_g}-{w_g}-prompt-{blip_prompts[i]}-{weather_prompts[i]}.png'
            gen_images[i].save(save_gen_img)  

if __name__ == '__main__':
    model_path = Path('ckpts/best_ckpts/sine_weather.pt')
    time = model_path.as_posix().split('/')[-2]

    model = GuidanceModel(init_g=50.0,
                          divide_out=0.1,
                          hidden_dim=768*2,
                          num_layers=5,
                          num_guidance_info=1,
                          lower_bound=1.0).to('cuda')

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    main(model,'sine')

'''if __name__ == '__main__':
    model_path = Path('ckpts/0416222403/47_model.pt')
    time = model_path.as_posix().split('/')[-2]
    #model_path = Path('ckpts/0326180211_model.pt')
    #time = model_path.stem.split('_')[0]
    model = AttentionModel(init_g=50.0,
                          divide_out=0.1,
                          hidden_dim=768,
                          num_layers=2,
                          head=4,
                          length=2,
                          num_guidance_info=1).to('cuda')
    model = MultiConditionAttentionModel(init_g=100.0,
                                         divide_out=0.2,
                                         num_layers=5,
                                         hidden_dim=768,
                                         heads=8).to('cuda')
    
    model = MultiConditionAttentionModel2(init_g=100.0,
                                         divide_out=0.2,
                                         num_layers=5,
                                         hidden_dim=768,
                                         heads=8).to('cuda')
                                         
    model = MultiConditionAttentionBLIPModel(init_g=50.0,
                                             init_blip_g=50.0,
                                             divide_out=0.1,
                                             num_layers=5,
                                             hidden_dim=768,
                                             heads=8).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    main(model, time, 'decrease')
    #multi_condition_main(model, time,'constant')
    #blip_main(model, time, 'decrease')'''