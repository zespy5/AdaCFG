from util.metric import Clip, Clip_ds,Clip_ds2, Dino, Clip_sturcture
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main(name):

    
    data_root = Path(f'check_valid/{name}')
    metric_dict={}
    total_clip = 0
    total_dino = 0
    total_neg_clip =0
    for _image in tqdm([*data_root.glob('*')]):
        image_name = _image.stem

        origin_image_path = _image/'origin.png'
        origin_image = Image.open(origin_image_path)
        for gen in _image.glob('g*'):
            gen_image = Image.open(gen)
            guidance = gen.stem.split('-')[1]
            to_prompt = gen.stem.split('-')[-1].replace('a photograph of a street','')
            
            _metrics = {'guidance': guidance}
            
            _metrics['image_name'] = image_name
            

            clip_score = Clip(gen_image, to_prompt)
            _metrics['clip'] = clip_score
            total_clip += clip_score
            clip_score = Clip(gen_image, 'ugly, blurry, low res, unrealistic, paint, distorttion, black and white')
            _metrics['negative clip'] = clip_score
            total_neg_clip += clip_score
            dino_score = Dino(origin_image, gen_image)
            _metrics['dino'] = dino_score
            total_dino +=dino_score
            metric_dict[gen.as_posix()] = _metrics
    instance = len(metric_dict)
    mean_clip = total_clip/instance
    mean_neg_clip = total_neg_clip/instance
    mean_dino = total_dino/instance
    print(mean_clip, mean_neg_clip, mean_dino)
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.index.name = 'file name'
    
    csv_path = data_root/f"{name}-metric_results.csv"
    df.to_csv(csv_path)
    
if __name__ == '__main__':
    main('val-0310114459')