import json
import os

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, List, Set, Tuple

BASE_PATH = 'C:/KU/ai-stylist/musinsa-en'

class MusinsaDataset(Dataset):
    def __init__(
        self,
        path = BASE_PATH,
    ):
        super().__init__()
        
        self.path = path
        self.top_items, self.bottom_items, self.shoes_items = self._get_item_dict()
        
    
    def __len__(self):
        return len(self.top_paths)
    
    def __getitem__(self, idx):
        return self.top_items[idx], self.bottom_items[idx], self.shoes_items[idx]
    
    def _get_item_dict(self):
        top_items = []
        bottom_items = []
        shoes_items = []
        
        with open(os.path.join(self.path, 'jsonfile.json'), "r") as f:
            data = json.load(f)
        
        # import pdb; pdb.set_trace()
        
        for item in list(data.keys()):
            top_items.append(data[item][0])
            bottom_items.append(data[item][1])
            shoes_items.append(data[item][2])
        
        return top_items, bottom_items, shoes_items
