from util.metric import Clip, Clip_ds,Clip_ds2, Dino, Clip_sturcture
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main():
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
    
    times = [' at night time',
                ' at sunset',
                ' at daytime']
    
    weathers = [' on a summer day',
                ' on a spring day',
                ' on a winter day',
                ' on an autumn day',
                ' on a rainy day',
                ' on a foggy day',
                ' on a snowy day',
                ' on a sunny day',
                ' on a cloudy day',]
    
    data_root = Path('experiments_results/multi-condition')
    from_prompt = 'a photograph of a street'
    metric_dict = {}
    for category in data_root.glob('*'):
        for _image in tqdm([*category.glob('*')]):
            image_name = _image.stem
            
            for time in times:
                time_path = _image/time
                for weather in weathers:
                    condition = time_path/weather
                    origin_image_path = condition/'origin.png'
                    origin_image = Image.open(origin_image_path)
                    for gen in condition.glob('g*'):
                        gen_image = Image.open(gen)
                        guidance = gen.stem.split('-')[1]
                        to_prompt = gen.stem.split('-')[-1]
                        
                        _metrics = {'guidance': guidance}
                        
                        _metrics['image_name'] = image_name
                        _metrics['time_condition'] = time
                        _metrics['weather_condition'] = weather
                        
                        time_to_prompt = from_prompt+time
                        weather_to_prompt = from_prompt+weather
                        

                        clip_score = Clip(gen_image, to_prompt)
                        _metrics['clip'] = clip_score
                        clip_score = Clip(gen_image, time_to_prompt)
                        _metrics['time clip'] = clip_score
                        clip_score = Clip(gen_image, weather_to_prompt)
                        _metrics['weather clip'] = clip_score
                        clip_score = Clip(gen_image, 'ugly, blurry, low res, unrealistic, paint, distorttion')
                        _metrics['negative clip'] = clip_score
                        
                        clip_ds_score = Clip_ds(origin_image, gen_image, from_prompt, to_prompt)
                        _metrics['clip ds'] = clip_ds_score
                        clip_ds_score = Clip_ds(origin_image, gen_image, from_prompt, time_to_prompt)
                        _metrics['time clip ds'] = clip_ds_score
                        clip_ds_score = Clip_ds(origin_image, gen_image, from_prompt, weather_to_prompt)
                        _metrics['weather clip ds'] = clip_ds_score
                        
                        clip_ds2_score, idx = Clip_ds2(times, origin_image, gen_image, from_prompt, time_to_prompt)
                        _metrics['time clip ds 2'] = clip_ds2_score
                        _metrics['clip ds 2 time from prompt'] = from_prompt+conditions[idx]
                        clip_ds2_score, idx = Clip_ds2(weathers, origin_image, gen_image, from_prompt, weather_to_prompt)
                        _metrics['weather clip ds 2'] = clip_ds2_score
                        _metrics['clip ds 2 time from prompt'] = from_prompt+conditions[idx]

                        
                        clip_structure_score = Clip_sturcture(origin_image, gen_image)
                        _metrics['clip structure'] = clip_structure_score
                        dino_score = Dino(origin_image, gen_image)
                        _metrics['dino'] = dino_score
                        
                        metric_dict[gen.as_posix()] = _metrics
                
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.index.name = 'file name'
    
    csv_path = data_root/"metric_results.csv"
    df.to_csv(csv_path)
    
if __name__ == '__main__':
    main()