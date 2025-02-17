from pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from util.guidance_scheduler import GuidanceScheduler
import torch


def main():    
    conditions = [' on a summer day',
                 ' on a spring day',
                 ' on a winter day',
                 ' on a autumn day',
                 ' on a rainy day',
                 ' on a foggy day',
                 ' on a snowy day',
                 ' on a sunny day',
                 ' on a cloudy day',
                 ' at night',
                 ' at sunset',
                 ' at daytime']
    
    pipe = PnPPipeline(generate_condition_prompt=True)
    #guidance_scheduler = GuidanceScheduler()
    
    save_root = Path('debug_results')
    save_root.mkdir(exist_ok=True)
    
    dirs = [*Path('image_data/train').glob('*')]
    
    '''tensor = torch.tensor([[50,0.00085,0.015],
                           [70,0.00085,0.012]])
    
    guidance_schedulers = guidance_scheduler.get_guidance_scales(tensor)'''
    
    conditions=[[' on a winter day',' on a autumn day',' on a summer day', ' on a spring day',],
                [' on a snowy day',' on a foggy day',' on a rainy day',' on a sunny day',],
                [' at daytime',' at sunset',' at night',' at daytime']]
    results = pipe(num_condition=3,
                   image_dirs=dirs[:4],
                   prompts=conditions,
                   #guidance_scales=guidance_schedulers,
                   negative_prompt='ugly, blurry, low res, unrealistic, paint')
    images = results.images

    for i in range(4):
        gen_file_name = save_root/f'{i}.png'
        images[i].resize((512,512)).save(gen_file_name)



if __name__ == "__main__": 
    main()
                
 
                
    
        
    
