import argparse
import torch
from pnp import PNP
from preprocess import run
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from pnp_utils import *


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        
def generate_prompt(images):
  
  text = "a photography of"
  inputs = processor(images, text, return_tensors="pt").to("cuda", torch.float16)

  out = model.generate(**inputs)
  caption = processor.decode(out[0], skip_special_tokens=True)
  if 'at night' in caption:
      caption = caption.split('at night')[0]

  return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/0.jpg')
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=999)
    parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    
    # general
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--prompt', type=str)
    # data
    parser.add_argument('--latents_path', type=str)
    # diffusion
    parser.add_argument('--guidance_scale', type=float, default=30)
    parser.add_argument('--n_timesteps', type=int, default=50)
    parser.add_argument('--negative_prompt', type=str, default='ugly, blurry, low res, unrealistic, paint')
    
    #injection threshold
    parser.add_argument('--pnp_attn_t', type=float, default=0.9)
    parser.add_argument('--pnp_f_t', type=float, default=0.9)
    
    opt = parser.parse_args()
    config = vars(parser.parse_args())
    
    pnp = PNP(config)
    
    save_root = Path('alpha_rev_scheduler_Nonecov_generate_results')
    save_root.mkdir(exist_ok=True)
    
    day_dir = [*Path('images_upright/day/milestone').glob('*')][:50]
    night_dir = [*Path('images_upright/night/nexus5x').glob('*')][:50]
    
    day_coditions = [' on a summer day',
                 ' on a spring day',
                 ' on a winter day',
                 ' on a autumn day',
                 ' on a rainy day',
                 ' on a foggy day',
                 ' on a snowy day',
                 ' on a sunny day',
                 ' on a cloudy day',
                 ' at night',
                 ' at sunset']
    night_coditions = [' on a summer day',
                 ' on a spring day',
                 ' on a winter day',
                 ' on a autumn day',
                 ' on a rainy day',
                 ' on a foggy day',
                 ' on a snowy day',
                 ' on a sunny day',
                 ' on a cloudy day',
                 ' at sunset',
                 ' at daytime']
    
    for dn in ['day', 'night']:
        conditions = day_coditions if dn=='day' else night_coditions
        image_p= day_dir if dn=='day' else night_dir
        save_time_dir = save_root/dn
        save_time_dir.mkdir(exist_ok=True)
        
        for cond in conditions:
            save_condition_dir = save_time_dir/cond
            save_condition_dir.mkdir(exist_ok=True)
            
            for i in range(50):
                config['image_path'] = image_p[i].as_posix()
                opt.data_dir = image_p[i].as_posix()
                origin_img = Image.open(config['image_path'])
                prompt = generate_prompt(origin_img)
                edited_prompt = prompt+cond
                
                save_gen_img_dir = save_condition_dir/f'{i}'
                save_gen_img_dir.mkdir(exist_ok=True)
                save_origin_img = save_gen_img_dir/f'{edited_prompt}.png'
                origin_img.resize((512,512)).save(save_origin_img)
                
                data_file = Path(opt.data_dir)
                latent_file = Path(opt.save_dir+'_forward')/data_file.stem/'noisy_latents_999.pt'
                if latent_file.exists():
                    print('There exist latent files')
                else:
                    run(opt)
                    
                config['latents_path'] = 'latents_forward/'+image_p[i].stem
                
                
                config['prompt'] = edited_prompt

                for g in [30.0, 50.0, 70.0, 100.0]:
                    config['guidance_scale']=g
                    pnp.reset_config(config)
                    gen_img = pnp.run()
                    gen_file_name = save_gen_img_dir/f'guid{int(g)}.png'
                    gen_img.resize((512,512)).save(gen_file_name)

                
 
                
    
        
    
