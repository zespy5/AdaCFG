from datetime import datetime
import yaml
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
i2t_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", 
                                                                      torch_dtype=torch.float16).to('cuda:0')

@torch.no_grad()
def generate_prompt(image, init_text='a photograph of'):
    inputs = image_processor(image,
                             init_text,
                             return_tensors="pt").to('cuda:0', torch.float16)

    outputs = i2t_model.generate(**inputs)

    caption = image_processor.decode(outputs[0], skip_special_tokens=True)

    return caption.rstrip()

def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config