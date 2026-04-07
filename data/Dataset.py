from typing import Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

        
class DomainChangeZeroShotDataset(Dataset):
    def __init__(self,
                 data_directory : Union[str, Path],
                 latents_path: Union[str, Path],
                 embedding_path: Union[str, Path],
                 data_length : int,
                 ):
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.image_names = sorted([*self.data_directory.glob('*')])[:data_length]
        assert len(self.image_names)!=0, "There is no image files"
        
        
        
        self.latents = torch.load(latents_path, weights_only=False, map_location='cpu')
        embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        self.image_embeddings = embeddings['image']
        self.text_embeddings = embeddings['text']
        
        conditions = list(self.text_embeddings.keys())
        
        self.conditions = conditions
        self.condition_length = len(conditions)
        self.data_length = data_length
        
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
        
        
        text_condition_embedding_infos = self.text_embeddings[condition]
        image_embedding_infos = self.image_embeddings[image_file_name]
        
        condition_mean = text_condition_embedding_infos['mean_embedding']
        prompt_emb_pair = text_condition_embedding_infos['prompt_emb_pair']
        
        prompts = list(prompt_emb_pair.keys())
        prompt_number = torch.randint(len(prompts), (1,))[0]
        prompt = prompts[prompt_number]
        tensors = prompt_emb_pair[prompt]
        
        sd_text_embedding = tensors['sd_clip']
        to_clip_embedding = tensors['clip']
        image_embedding = image_embedding_infos['image_project_embedding'].squeeze()
        return (idx,
                prompt_number,
                condition_number,
                real_image_tensor,
                image_embedding,
                condition_mean,
                latents,
                sd_text_embedding,
                to_clip_embedding)


class DomainChangeIP2PDataset(Dataset):
    def __init__(self,
                 data_directory : Union[str, Path],
                 embedding_path: Union[str, Path],
                 data_length : int
                 ):
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.image_names = sorted([*self.data_directory.glob('*')])[:data_length]
        assert len(self.image_names)!=0, "There is no image files"
        
        
        embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        self.image_embeddings = embeddings['image']
        self.text_embeddings = embeddings['text']
        
        conditions = list(self.text_embeddings.keys())
        
        self.conditions = conditions
        self.condition_length = len(conditions)
        self.data_length = data_length
        
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
        
        
        
        text_condition_embedding_infos = self.text_embeddings[condition]
        image_embedding_infos = self.image_embeddings[image_file_name]
        
        condition_mean = text_condition_embedding_infos['mean_embedding']
        prompt_emb_pair = text_condition_embedding_infos['prompt_emb_pair']
        
        prompts = list(prompt_emb_pair.keys())
        prompt_number = torch.randint(len(prompts), (1,))[0]
        prompt = prompts[prompt_number]
        tensors = prompt_emb_pair[prompt]
        
        sd_text_embedding = tensors['sd_clip']
        to_clip_embedding = tensors['clip']
        image_embedding = image_embedding_infos['image_project_embedding'].squeeze()
        return (idx,
                prompt_number,
                condition_number,
                real_image_tensor,
                image_embedding,
                condition_mean,
                sd_text_embedding,
                to_clip_embedding)