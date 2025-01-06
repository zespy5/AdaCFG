from PIL import Image
import numpy as np
from pathlib import Path
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    return image

dataset_path = Path('images_upright')
gen_path = Path('none_scheduler_none_rescale_generate_results')
r_gen_path = Path('none_scheduler_rescale_generate_results')
s_gen_path = Path('alpha_rev_scheduler_generate_results')
sr_gen_path = Path('scheduler_rescale_generate_results')
print('start')
image_paths = [*dataset_path.glob('*/*/*')]
ori_paths = [*gen_path.glob('*/*/*/a photo*')]
gen30_paths = [*gen_path.glob('*/*/*/*30.png')]
gen50_paths = [*gen_path.glob('*/*/*/*50.png')]
gen70_paths = [*gen_path.glob('*/*/*/*70.png')]
gen100_paths = [*gen_path.glob('*/*/*/*100.png')]

r_gen30_paths = [*r_gen_path.glob('*/*/*/*30.png')]
r_gen50_paths = [*r_gen_path.glob('*/*/*/*50.png')]
r_gen70_paths = [*r_gen_path.glob('*/*/*/*70.png')]
r_gen100_paths = [*r_gen_path.glob('*/*/*/*100.png')]

s_gen30_paths = [*s_gen_path.glob('*/*/*/*30.png')]
s_gen50_paths = [*s_gen_path.glob('*/*/*/*50.png')]
s_gen70_paths = [*s_gen_path.glob('*/*/*/*70.png')]
s_gen100_paths = [*s_gen_path.glob('*/*/*/*100.png')]

sr_gen30_paths = [*sr_gen_path.glob('*/*/*/*30.png')]
sr_gen50_paths = [*sr_gen_path.glob('*/*/*/*50.png')]
sr_gen70_paths = [*sr_gen_path.glob('*/*/*/*70.png')]
sr_gen100_paths = [*sr_gen_path.glob('*/*/*/*100.png')]


real_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(image_paths)]
ori_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(ori_paths)]
gen30_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(gen30_paths)]
gen50_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(gen50_paths)]
gen70_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(gen70_paths)]
gen100_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(gen100_paths)]

r_gen30_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(r_gen30_paths)]
r_gen50_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(r_gen50_paths)]
r_gen70_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(r_gen70_paths)]
r_gen100_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(r_gen100_paths)]

s_gen30_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(s_gen30_paths)]
s_gen50_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(s_gen50_paths)]
s_gen70_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(s_gen70_paths)]
s_gen100_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(s_gen100_paths)]

sr_gen30_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(sr_gen30_paths)]
sr_gen50_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(sr_gen50_paths)]
sr_gen70_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(sr_gen70_paths)]
sr_gen100_images = [np.array(Image.open(path).convert('RGB').resize((512,512))) for path in tqdm(sr_gen100_paths)]



print('data load')

real_images = torch.cat([preprocess_image(image) for image in real_images])
ori_images = torch.cat([preprocess_image(image) for image in ori_images])
gen30_images = torch.cat([preprocess_image(image) for image in tqdm(gen30_images)])
gen50_images = torch.cat([preprocess_image(image) for image in tqdm(gen50_images)])
gen70_images = torch.cat([preprocess_image(image) for image in tqdm(gen70_images)])
gen100_images = torch.cat([preprocess_image(image) for image in tqdm(gen100_images)])

r_gen30_images = torch.cat([preprocess_image(image) for image in tqdm(r_gen30_images)])
r_gen50_images = torch.cat([preprocess_image(image) for image in tqdm(r_gen50_images)])
r_gen70_images = torch.cat([preprocess_image(image) for image in tqdm(r_gen70_images)])
r_gen100_images = torch.cat([preprocess_image(image) for image in tqdm(r_gen100_images)])

s_gen30_images = torch.cat([preprocess_image(image) for image in tqdm(s_gen30_images)])
s_gen50_images = torch.cat([preprocess_image(image) for image in tqdm(s_gen50_images)])
s_gen70_images = torch.cat([preprocess_image(image) for image in tqdm(s_gen70_images)])
s_gen100_images = torch.cat([preprocess_image(image) for image in tqdm(s_gen100_images)])

sr_gen30_images = torch.cat([preprocess_image(image) for image in tqdm(sr_gen30_images)])
sr_gen50_images = torch.cat([preprocess_image(image) for image in tqdm(sr_gen50_images)])
sr_gen70_images = torch.cat([preprocess_image(image) for image in tqdm(sr_gen70_images)])
sr_gen100_images = torch.cat([preprocess_image(image) for image in tqdm(sr_gen100_images)])





print('convert torch')
gen_datas = [ori_images, gen30_images, gen50_images, gen70_images, gen100_images,
             r_gen30_images, r_gen50_images, r_gen70_images, r_gen100_images,
             s_gen30_images, s_gen50_images, s_gen70_images, s_gen100_images,
             sr_gen30_images, sr_gen50_images, sr_gen70_images, sr_gen100_images]

fids = []

fid = FrechetInceptionDistance(reset_real_features=False).to('cuda')
real_images = real_images.to('cuda')
fid.update(real_images, real=True)
for gens in gen_datas:
    gens = gens.to('cuda')
    fid.update(gens, real=False)
    co= float(fid.compute())
    print(co)
    fids.append(co)
    fid.reset()
    
    
print(fids)
    