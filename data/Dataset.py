from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPModel, CLIPProcessor
from tqdm import tqdm
from torchvision.transforms import ToTensor
from random import choice
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


class DomainMultiChangeDataset(Dataset):
    def __init__(self,
                 data_directory : Union[str, Path],
                 latents_path: Union[str, Path],
                 embedding_path: Union[str, Path],
                 data_length : int,
                 time_conditions : List[str],
                 time_conditions_embedding: torch.Tensor,
                 weather_conditions : List[str],
                 weather_conditions_embedding: torch.Tensor
                 ):
        self.data_directory = Path(data_directory) if isinstance(data_directory, str) else data_directory
        self.image_names = sorted([*self.data_directory.glob('*')])[:data_length]
        assert len(self.image_names)!=0, "There is no image files"
        
        self.time_conditions = time_conditions
        self.time_condition_length = len(time_conditions)
        self.time_conditions_embedding = time_conditions_embedding
        
        self.weather_conditions = weather_conditions
        self.weather_condition_length = len(weather_conditions)
        self.weather_conditions_embedding = weather_conditions_embedding
        
        self.data_length = data_length
        
        self.latents = torch.load(latents_path, weights_only=False, map_location='cpu')
        self.embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        
        self.transform = ToTensor()
        
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        
        time_condition_number = torch.randint(self.time_condition_length, (1,))[0]
        time_condition = self.time_conditions[time_condition_number]
        weather_condition_number = torch.randint(self.weather_condition_length, (1,))[0]
        weather_condition = self.weather_conditions[weather_condition_number]
        
        image_dir = self.image_names[idx]
        real_image = Image.open(image_dir).convert('RGB').resize((512,512))
        real_image_tensor = self.transform(real_image)
        
        image_file_name = image_dir.stem
        
        latents = self.latents[image_file_name]
        
        embedding_infos = self.embeddings[image_file_name]
        
        time_sd_text_embedding = embedding_infos['sd_text_embeddings'][time_condition].squeeze()
        weather_sd_text_embedding = embedding_infos['sd_text_embeddings'][weather_condition].squeeze()
        
        model_time_input_embedding = self.time_conditions_embedding[time_condition_number].squeeze()
        model_weather_input_embedding = self.weather_conditions_embedding[weather_condition_number].squeeze()
        
        image_embedding = embedding_infos['image_project_embedding'].squeeze()
        
        from_clip_embedding = embedding_infos['text_project_embeddings']['origin'].squeeze()
        to_time_clip_embedding = embedding_infos['text_project_embeddings'][time_condition].squeeze()
        to_weather_clip_embedding = embedding_infos['text_project_embeddings'][weather_condition].squeeze()
        idx = torch.tensor(idx, dtype=torch.int16)
        return (idx,
                time_condition_number,
                weather_condition_number,
                real_image_tensor,
                image_embedding,
                model_time_input_embedding,
                model_weather_input_embedding,
                latents,
                time_sd_text_embedding,
                weather_sd_text_embedding,
                from_clip_embedding,
                to_time_clip_embedding,
                to_weather_clip_embedding)



class DomainMultiChangeBLIPDataset(Dataset):
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
        
        blip_sd_text_embedding = embedding_infos['sd_text_embeddings']['origin'].squeeze()
        condition_sd_text_embedding = embedding_infos['sd_text_embeddings'][condition].squeeze()
        
        image_embedding = embedding_infos['image_project_embedding'].squeeze()
        
        from_clip_embedding = embedding_infos['text_project_embeddings']['origin'].squeeze()
        to_clip_embedding = embedding_infos['text_project_embeddings'][condition].squeeze()
        idx = torch.tensor(idx, dtype=torch.int16)
        return (idx,
                condition_number,
                real_image_tensor,
                image_embedding,
                latents,
                blip_sd_text_embedding,
                condition_sd_text_embedding,
                from_clip_embedding,
                to_clip_embedding)
        
class DomainChangeZeroShotDataset(Dataset):
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
        embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        self.image_embeddings = embeddings['image']
        self.text_embeddings = embeddings['text']
        
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

class DomainChangeBLIPZeroShotDataset(Dataset):
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
        embeddings = torch.load(embedding_path, weights_only=False, map_location='cpu')
        self.image_embeddings = embeddings['image']
        self.text_embeddings = embeddings['text']
        
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
        blip_sd_embedding = image_embedding_infos['sd_clip'].squeeze()
        blip_clip_embedding = image_embedding_infos['clip'].squeeze()
        image_idx = torch.tensor(idx, dtype=torch.int16)
        prompt_idx = torch.tensor(prompt_number, dtype=torch.int16)
        return (image_idx,
                condition_number,
                real_image_tensor,
                image_embedding,
                condition_mean,
                latents,
                sd_text_embedding,
                to_clip_embedding,
                blip_sd_embedding,
                blip_clip_embedding)