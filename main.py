from pnp import PnPPipeline
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from pnp_utils import *




if __name__ == "__main__":    
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
    
    pipe = PnPPipeline()
    
    save_root = Path('check_generate_results')
    save_root.mkdir(exist_ok=True)
    
    dirs = [*Path('images_upright/day/milestone').glob('*')]

    results = pipe(image_dirs=dirs[:2],
                   conditions=conditions,
                   latents_save_root='latents_forward2',
                   negative_prompt='ugly, blurry, low res, unrealistic, paint')
    images = results.images
    prompts = results.prompts
    for i in range(2):
        gen_file_name = save_root/f'{prompts[i]}.png'
        images[i].resize((512,512)).save(gen_file_name)


                
 
                
    
        
    
