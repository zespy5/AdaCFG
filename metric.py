from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

def dino_score(real, gen):
    with torch.no_grad():
        inputs1 = processor(images=real, return_tensors="pt").to('cuda')
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.view(-1)
        
        inputs2 = processor(images=gen, return_tensors="pt").to('cuda')
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.view(-1)
        
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1,image_features2).item()
    sim = (sim+1)/2

    return sim

    

#fid = FrechetInceptionDistance(reset_real_features=False).to('cuda')
clip_score = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14', ).to('cuda')
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to('cuda')
#ssim = StructuralSimilarityIndexMeasure().to('cuda')

save_root = Path('metric_results2')
save_root.mkdir(exist_ok=True)


dataset_path = Path('images_upright')
gen_path = [Path('none_scheduler_none_rescale_generate_results'),
            Path('scheduler_none_rescale_generate_results'),
            Path('none_scheduler_rescale_generate_results'),
            Path('scheduler_rescale_generate_results')]

for paths in gen_path:
    filepath = paths/'night'/' at daytime'/'0'
    guidances = sorted([int(g.stem.split('-')[0]) for g in filepath.glob('*.pt')])
    print([*filepath.glob('*.pt')])

    #fid_data = {}
    clip_score_value = {}
    dino_score_value = {}
    #ssim_value = {}
    clip_score_value['file_name'] = []
    dino_score_value['file_name'] = []
    #ssim_value['file_name'] = []
        

    for g in guidances:
        #fid_data[g] =[]
        clip_score_value[str(g)]=[]
        dino_score_value[str(g)]=[]
        #ssim_value[str(g)]=[]
        

    for dn in paths.glob('*'):
        for cond in dn.glob('*'):
            for img in cond.glob('*'):
                clip_score_value['file_name'].append(img.as_posix())
                dino_score_value['file_name'].append(img.as_posix())
                #ssim_value['file_name'].append(img.as_posix())
                for gui in guidances:
                    prompt_dir = [*img.glob('a photo*')]
                    prompt = prompt_dir[0].stem
                    
                    real_image = Image.open(prompt_dir[0])
                    gen_image_file_name = img/f'guid{gui}.png'
                    gen_image = Image.open(gen_image_file_name)
                    
                    dino_similarity = dino_score(real_image, gen_image)
                    dino_score_value[str(gui)].append(dino_similarity)
                    
                    real_img = np.array(real_image.convert('RGB').resize((512,512)))
                    image = np.array(gen_image.convert('RGB').resize((512,512)))
                    real_image = preprocess_image(real_img)
                    image = preprocess_image(image)
                    
                    #ssim_score = ssim((real_image/255.0).to('cuda'), (image/255.0).to('cuda'))
                    #ssim_value[str(gui)].append(ssim_score.item())

                    #fid_data[gui].append(image)
                    cs=clip_score(image.to('cuda'), prompt)
                    clip_score_value[str(gui)].append(cs.item())

                    
    clip_df = pd.DataFrame(clip_score_value)
    dino_df = pd.DataFrame(dino_score_value)
    #ssim_df = pd.DataFrame(ssim_value)
    
    save_clip_df = save_root/f'clip_{paths.stem}.csv'
    save_dino_df = save_root/f'dino_{paths.stem}.csv'
    #save_ssim_df = save_root/f'ssim_{paths.stem}.csv'
    
    clip_df.to_csv(save_clip_df)
    dino_df.to_csv(save_dino_df)
    #ssim_df.to_csv(save_ssim_df)
    

'''del clip_score
del model

image_paths = [*dataset_path.glob('*/*/*')]

real_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(image_paths)]
real_images = torch.cat([preprocess_image(image) for image in tqdm(real_images)])

print('FID')
real_images = real_images.to('cuda')
fid.update(real_images, real=True)
for g in guidances:
    gens = torch.cat(fid_data[g])
    gens = gens.to('cuda')
    fid.update(gens, real=False)
    co= fid.compute().detach().item()
    print(f'gudiance {g} :', end=' ')
    print(round(co,2))
    fid.reset()'''
    

    