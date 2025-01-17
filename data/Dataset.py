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
                 device : str = "cuda",

                 ):
        
        self.device = device
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        
        self.image_names = [*self.data_directory.glob('*')]
        assert len(self.image_names)==0, "There is no image files"
        
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_file_name = self.image_names[idx]
        real_image = Image.open(image_file_name).convert('RGB')
        
        
        return image_file_name, real_image
    
