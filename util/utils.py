from datetime import datetime
import yaml
import torch
import json

def get_timestamp(date_format: str = "%m%d%H%M%S") -> str:
    timestamp = datetime.now()
    return timestamp.strftime(date_format)


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_json(config_path):
    with open(config_path, "r",encoding="euc-kr", errors="replace") as f:
        config = json.load(f)
        
    return config