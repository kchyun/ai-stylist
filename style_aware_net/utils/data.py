import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Optional, Dict, List, Set, Tuple

from PIL import Image


class StyleAwareNetDataset(Dataset):
    def __init__(
        self,
        rootdir,
        is_train,
        transform=None
        ):
        super(StyleAwareNetDataset, self).__init__()
        self.pairs_path = os.path.join(
            rootdir, 'data', 'polyvore_cleaned', 
            'train.json' if is_train else 'valid.json')
        self.top_ids_path = os.path.join(
            rootdir, 'data', 'polyvore_cleaned',
            'top_embeds.json')
        self.bottom_ids_path = os.path.join(
            rootdir, 'data', 'polyvore_cleaned',
            'bottom_embeds.json')
        self.img_path = os.path.join(
            rootdir, 'data', 'polyvore_outfits', 'images')

        self.transform = transform

        self.pos_pairs, self.neg_check = self._preprocess_pairs(self.pairs_path)
        self.top_ids = pd.read_json(self.top_ids_path).index
        self.bottom_ids = pd.read_json(self.bottom_ids_path).index
        

    def __len__(self):
        return len(self.pos_pairs)
    

    def __getitem__(self, idx):
        anc_id, pos_id = self.pos_pairs[idx]
        neg_ids = self._get_neg_ids(anc_id, n_sample=8)

        anc_img = self._id2img(anc_id)
        pos_img = self._id2img(pos_id)
        neg_imgs = torch.stack([self._id2img(neg_id) for neg_id in neg_ids])
        return anc_img, pos_img, neg_imgs
    

    def _id2img(self, img_id):
        img = Image.open(os.path.join(self.img_path, f'{img_id}.jpg'))
        img = np.array(img.convert("RGB"))[:, :, ::-1]
        if self.transform:
            img = self.transform(image=img)['image']
        return torch.from_numpy(img.copy())


    def _preprocess_pairs(self, path):
        df = pd.read_json(path)\
            .drop_duplicates(subset=['top_id', 'bottom_id'])\
            .groupby(['top_id', 'y'])\
            .agg({'bottom_id' : set})\
            .xs(1, level='y')
        
        neg_check = df
        pos_pairs = df.explode('bottom_id').reset_index().to_numpy()

        return pos_pairs, neg_check
    

    def _get_neg_ids(self, anc_id, n_sample=8):
        neg_ids = set()
        while len(neg_ids) < n_sample:
            neg_id = random.choice(self.bottom_ids)
            if neg_id not in self.neg_check.loc[anc_id]['bottom_id']:
                neg_ids.add(neg_id)
        return list(neg_ids)
    

def get_dataset(transform):
    BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

    train_dataset = StyleAwareNetDataset(
        BASE_PATH, is_train=True, transform=transform)
    valid_dataset = StyleAwareNetDataset(
        BASE_PATH, is_train=False, transform=None)
    
    return train_dataset, valid_dataset

if __name__ == '__main__':
    import albumentations as A

    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.Rotate(limit=15),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(0, 1)
        ])
    
    t, v= get_dataset(transform)
    print(t[0])