from util.metric import Clip, Clip_txt_mean, Clip_txt_mean_sim, Dino, Clip_sturcture
from util.utils import get_json
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main(name):

    large_conditions = get_json('configs/large_conditions.json')
    small_conditions = get_json('configs/conditions.json')
    data_root = Path(f'check_valid/{name}')
    metric_dict={}
    total_clip = 0
    total_dino = 0
    small_conditions_key = list(small_conditions.keys())
    large_conditions_key = list(large_conditions.keys())
    condition_change = {}
    for condition in small_conditions_key:
        for large_condition in large_conditions_key:
            if large_condition in condition:
                condition_change[large_condition] = condition
                break

    conditions_emb = {}
    for con in large_conditions.keys():
        prompts = large_conditions[con]
        conditioned_embeddings = Clip_txt_mean(prompts)
        conditions_emb[condition_change[con]] = conditioned_embeddings

    for _image in tqdm([*data_root.glob('*')]):
        image_name = _image.stem

        origin_image_path = _image/'origin.png'
        origin_image = Image.open(origin_image_path)
        for category in _image.glob('*'):
            cat = category.stem
            for gen in category.glob('*'):
                gen_image = Image.open(gen)
                guidance = float(gen.stem.split('-')[1].replace('_','.'))
                to_prompt = f" a photo of a {cat} street "
                positive_prompt = f"{cat}, city street, well-structured, colourful, realistic, high resolution"
                negative_prompt ='ugly, blurry, low resolution, unrealistic, paint, distorttion, black and white photograph'
                _metrics = {'guidance': guidance}
                
                _metrics['image_name'] = image_name
                
                clip_score = Clip(gen_image, to_prompt)
                _metrics['clip'] = clip_score
                
                m_clip_score = Clip_txt_mean_sim(conditions_emb[cat], gen_image)
                _metrics['mean clip'] = m_clip_score
                total_clip += clip_score
                
                p_clip_score = Clip(gen_image, positive_prompt)
                _metrics['positive clip'] = p_clip_score
                
                n_clip_score = Clip(gen_image, negative_prompt)
                _metrics['negative clip'] = n_clip_score

                clip_iqa = p_clip_score/(p_clip_score+n_clip_score)
                _metrics['clip iqa'] = clip_iqa
                 
                dino_score = Dino(origin_image, gen_image)
                _metrics['dino'] = dino_score
                total_dino +=dino_score
                metric_dict[gen.as_posix()] = _metrics
    instance = len(metric_dict)
    mean_clip = total_clip/instance
    mean_dino = total_dino/instance
    print(mean_clip, mean_dino)
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.index.name = 'file name'
    
    csv_path = data_root/f"{name}-metric_results.csv"
    df.to_csv(csv_path)
    
if __name__ == '__main__':
    main('val-zero-shot-0416222403')
    #main('val-zero-shot-0416222444')