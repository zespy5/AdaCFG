from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor
from tqdm import tqdm
from torchvision.transforms import ToTensor
class DomainChangeDataset(Dataset):
    def __init__(self,
                 data_directory : Union[str, Path],
                 latents_path: Union[str, Path],
                 embedding_path: Union[str, Path],
                 data_length : int,
                 conditions : List[str],
                 conditions_embedding: torch.Tensor
                 ):
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.image_names = sorted([*self.data_directory.glob('*')])[:data_length]
        assert len(self.image_names)!=0, "There is no image files"
        
        self.conditions = conditions
        self.condition_length = len(conditions)
        self.conditions_embedding = conditions_embedding
        self.data_length = data_length
        
        self.latents = torch.load(latents_path, weights_only=False, map_location='cpu')
        self.embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        
        self.transform = ToTensor()
        
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        
        condition_number = torch.randint(self.condition_length, (1,))[0]
        condition = self.conditions[condition_number]
        image_dir = self.image_names[idx]
        real_image = Image.open(image_dir).convert('RGB').resize((512,512))
        real_image_tensor = self.transform(real_image)
        
        image_file_name = image_dir.stem
        
        latents = self.latents[image_file_name]
        
        embedding_infos = self.embeddings[image_file_name]
        
        sd_text_embedding = embedding_infos['sd_text_embeddings'][condition].squeeze()
        
        model_input_embedding = self.conditions_embedding[condition_number].squeeze()
        image_embedding = embedding_infos['image_project_embedding'].squeeze()
        from_clip_embedding = embedding_infos['text_project_embeddings']['origin'].squeeze()
        to_clip_embedding = embedding_infos['text_project_embeddings'][condition].squeeze()
        idx = torch.tensor(idx, dtype=torch.int16)
        return (idx,
                condition_number,
                real_image_tensor,
                image_embedding,
                model_input_embedding,
                latents,
                sd_text_embedding,
                from_clip_embedding,
                to_clip_embedding)
