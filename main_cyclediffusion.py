import torch
from PIL import Image
from pathlib import Path
from diffusers import DDIMScheduler
from cyclediffusion_hack import CycleDiffusionPipelineGuidance
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from models.model import *
from util.guidance_scheduler import GuidanceScheduler
from util.utils import get_json
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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


# make sure you're logged in with `huggingface-cli login`
model_id_or_path = "stabilityai/stable-diffusion-2-1-base"
scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
pipe = CycleDiffusionPipelineGuidance.from_pretrained(model_id_or_path, scheduler=scheduler).to("cuda")


categorys = {'clear day':'a photo of a street on a clear day',
             'cloudy day':'a photo of a street on a cloudy day',
             'foggy day':'a photo of a street on a foggy day',
             'rainy day':'a photo of a street on a rainy day',
             'snowy day':'a photo of a street on a snowy day',
             'night':'a photo of a street at night',
             'sunset':'a photo of a street at sunset'}
conditions = get_json('configs/conditions.json')

data_root = Path('image_data/eval')
origin_images = sorted([*data_root.glob('*.jpg')])
save_root = Path('check_valid/cycle-diffusion')
save_root.mkdir(exist_ok=True)
for c in categorys:
    category_root = save_root/categorys[c]
    category_root.mkdir(exist_ok=True)

strength = 0.5
num_inference_steps = 100
guidance_inference_step = strength*num_inference_steps

guidance_scheduler = GuidanceScheduler(gradient='decrease', device='cuda', 
                                       n_timestep=guidance_inference_step)

model_path = Path('ckpts/0610121259/18_model.pt')

model = GuidanceModel(init_g=100.0,
                        divide_out=0.05,
                        hidden_dim=768*2,
                        num_layers=8,
                        num_guidance_info=2).to('cuda')

model.load_state_dict(torch.load(model_path))
model.eval()

for o_img in tqdm(origin_images):
    init_image = Image.open(o_img).convert("RGB")
    source_prompt = 'a photo of a street'
    image_embedding = image_clip_embeds(init_image)
    
    for prompt in categorys.values():
        
        to_clip_embedding = prompt_embeds(prompt)
        
        model_input = torch.cat([image_embedding, to_clip_embedding], dim=1)
        guidance_value, velocity = model(model_input)

        guidance = guidance_scheduler.get_guidance_scales(guidance_value, velocity)

        # call the pipeline
        image = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            eta=0.005,
            strength=strength,
            guidance_scale=guidance,
            source_guidance_scale=1
        ).images[0]
        save_name = save_root/prompt/f'{o_img.stem}.png'
        image.save(save_name)


