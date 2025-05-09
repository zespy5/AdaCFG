from util.guidance_scheduler import GuidanceScheduler
from pnp import PnPPipeline
from models.model import *
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image
import numpy
from torch.nn.functional import cosine_similarity
from util.metric import (Clip, Dino, 
                         Clip_txt_mean, Clip_txt_mean_sim, 
                         prompt_embeds, image_clip_embeds)
from util.utils import get_json
import pandas as pd

@torch.no_grad()
def main(model, 
         domains, 
         grad, 
         conditions,
         image_save_root, 
         latent_save_root,
         pnp_rate=0.9,
         image_size=256): 
    
    from_domain = domains.split('2')[0]
    to_domain = domains.split('2')[1]
    
    save_root = Path('check_valid')
    save_root.mkdir(exist_ok=True)

    save_valid = save_root/domains
    save_valid.mkdir(exist_ok=True)
    

    pnp = PnPPipeline(pnp_attn_t=pnp_rate,
                      pnp_f_t=pnp_rate,
                      image_size=image_size)
    guidance_scheduler = GuidanceScheduler(gradient=grad)
    
    data_root = Path(image_save_root)
    image_dirs = sorted([*data_root.glob('*')])
    
    mean_embedding = Clip_txt_mean(conditions)
    positive_prompt = f"a photo of a {to_domain}"
    negative_prompt = f'ugly, blurry, low resolution, unrealistic, paint, distortion, black and white photograph, {from_domain}e'
    metric_dict={}
    
    for img_dir in tqdm(image_dirs):
        image_name = img_dir.stem
        
        save_images = save_valid/image_name
        save_images.mkdir(exist_ok=True)
        
        save_candidates = save_images/'candidates'
        save_candidates.mkdir(exist_ok=True)
        
        save_origin_image = save_images/'origin.png'
        origin_image = Image.open(img_dir).resize((image_size, image_size))
        origin_image.save(save_origin_image)
        image_embedding = image_clip_embeds(origin_image)
        

        batch_size = len(conditions)
        
        to_clip_embedding = prompt_embeds(conditions)
        img_embs = image_embedding.repeat(batch_size,1)

        model_input = torch.cat([img_embs,
                                to_clip_embedding], dim=1).view(batch_size, 2, -1)
        guidance_value = model(model_input)
        guidance = guidance_scheduler.get_guidance_scales(guidance_value)
        
        images = [img_dir]*batch_size
        outputs = pnp(image_dirs=images,
                    negative_prompt=f'ugly, blurry, low resolution, unrealistic, paint, distortion, {from_domain}',
                    prompts=[conditions],
                    guidance_scales=guidance,
                    latents_save_root=latent_save_root)
        
        guidance_values = guidance_value.squeeze().cpu().numpy()
        gen_images = outputs.images
        prompts = outputs.prompts[0]

        losses = []
        for i in range(batch_size):
            m_clip = Clip_txt_mean_sim(mean_embedding, gen_images[i])
            p_clip = Clip(gen_images[i], positive_prompt)
            n_clip = Clip(gen_images[i], negative_prompt)
            dino = Dino(origin_image, gen_images[i])
            loss = (1-m_clip)*0.5 + (1-p_clip)*0.5 + n_clip + (1-dino)*0.1
            g = guidance_values.item(i)
            save_gen_img = save_candidates/f'prompt-{prompts[i]}.png'
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
        save_picked_image = save_images/f'best_pick.png'
        min_image.save(save_picked_image)
        
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.index.name = 'file_name'
    df.to_csv(save_valid/f'{domains}_metrics.csv')

    
if __name__ == '__main__':
    model_path = Path('ckpts/best_ckpts/horse2zebra.pt')
    domains = model_path.stem
    from_domain = domains.split('2')[0]
    to_domain = domains.split('2')[1]
    
    
    model = AttentionModel(init_g=100.0,
                          divide_out=0.1,
                          hidden_dim=768,
                          num_layers=2,
                          head=4,
                          length=2,
                          num_guidance_info=1).to('cuda')

    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    conditions = get_json(f'configs/zebra_prompts.json')
    prompts = conditions[to_domain]
    main(model=model, 
         domains=domains, 
         grad='decrease',
         conditions= prompts,
         image_save_root=f'unpaired_image_data/horse_zebra/test_{from_domain}',
         latent_save_root=f'unpaired_image_data/horse_zebra_latents/test_{from_domain}_latents_forward',
         pnp_rate=0.9,
         image_size=256)
    
