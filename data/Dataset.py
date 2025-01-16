from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class DomainChangeDataset(Dataset):
    def __init__(self,
                 data_directory : Union[str, Path],
                 conditions : Optional[Union[str, List[str]]] = None,
                 device : str = "cuda",
                 init_text :str = "a photography of"
                 ):
        
        self.device = device
        self.init_text = init_text
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        
        if conditions is None:
            self.conditions = [""]
        else:
            self.conditions = [conditions] if isinstance(conditions,str) else conditions
        self.num_conditions = len(conditions)
        
        self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.i2t_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", 
                                                                      torch_dtype=torch.float16).to(self.device)

        self.image_names = [*self.data_directory.glob('*')]
        assert len(self.image_names)==0, "There is no image files"
        
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_file_name = self.image_names[idx]
        prompts = self.generate_prompt(idx)
        random_condition = np.random.randint(0, self.num_conditions)
        prompts += self.conditions[random_condition]
        
        return image_file_name, prompts
    
    def generate_prompt(self, idx):
        
        images = Image.open(self.image_names[idx]).convert('RGB')
        
        inputs = self.image_processor(images, self.init_text, 
                                      return_tensors="pt").to(self.device, torch.float16)

        outputs = self.i2t_model.generate(**inputs)

        captions = self.image_processor.decode(outputs[0], skip_special_tokens=True)
        captions = captions.replace(' at night','').rstrip()

        return captions