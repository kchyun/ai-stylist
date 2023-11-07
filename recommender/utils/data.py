import json
import random
from collections import OrderedDict, defaultdict

import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Optional, Dict, List, Set, Tuple

PAIRS_PATH = 'C:/Users/omniv/PycharmProjects/Recommendation/FashionVCdata/FashionVCdata/gnd_top_bottom_pairs.csv'
TOP_EMBEDS_PATH = 'C:/Users/omniv/PycharmProjects/Recommendation/FashionVCdata/FashionVCdata/top_embeds.json'
BOTTOM_EMBEDS_PATH = 'C:/Users/omniv/PycharmProjects/Recommendation/FashionVCdata/FashionVCdata/bottom_embeds.json'


class FashionMLPDataset(Dataset):
    def __init__(
        self,
        pairs: Dict[str, Set[str]],
        dataset_type: str='train'
    ):
        super(FashionMLPDataset).__init__()
        self.dataset_type = dataset_type
        self.pairs = pairs
        self.data = [(top, bottom) for top in self.pairs.keys() for bottom in self.pairs.get(top)]
        self.top_embeds = self._preprocess_embeds(TOP_EMBEDS_PATH)
        self.bottom_embeds = self._preprocess_embeds(BOTTOM_EMBEDS_PATH)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset_type == 'train':
            top_idx, bottom_idx = self.data[idx]
            neg_bottom_idx = self._get_neg_bottom(top_idx)
            positive_pair = (self.top_embeds[top_idx], self.bottom_embeds[bottom_idx])
            negative_pair = (self.top_embeds[top_idx], self.bottom_embeds[neg_bottom_idx])
            item = {'positive_pair' : positive_pair,
                    'negative_pair' : negative_pair}
        else:
            top_idx, bottom_idx = self.data[idx]
            item = {'input_embed' : self.top_embeds[top_idx],
                    'target' : bottom_idx}
            
        return item
            
    def _get_neg_bottom(self, top_idx) -> str:
        neg_bottom_idx = self.data[random.randint(0, len(self.data) - 1)][-1]
        cnt = 0
        while (neg_bottom_idx in self.pairs[top_idx]) & (cnt < 32):
            neg_bottom_idx = self.data[random.randint(0, len(self.data) - 1)][-1]
        return str(neg_bottom_idx)
        
    def _preprocess_embeds(self, path):
        with open(path, 'r') as f:
            items = json.load(f)
        return {str(item['id']): Tensor(item['embed']) for item in items}


def get_dataset() -> Tuple[FashionMLPDataset, FashionMLPDataset]:
    print('Dataset generating starts...')
    pair_df = pd.read_csv(PAIRS_PATH)
    pairs = defaultdict(list)
    for i, (top, bottom) in pair_df.iterrows():
        pairs[str(top)].append(str(bottom))
    train_pairs, test_pairs = _hit_rate_split(pairs)
    return (FashionMLPDataset(train_pairs, 'train'),
            FashionMLPDataset(test_pairs, 'test'))
    
    
def _hit_rate_split(pairs: Dict[str, List[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    print('Train, test spliting starts...')
    train_pairs, test_pairs = dict(), dict()
    for top_idx, bottom_list in pairs.items():
        test_bottom = bottom_list.pop()
        train_pairs[top_idx] = set(bottom_list)
        test_pairs[top_idx] = set(test_bottom)
    return train_pairs, test_pairs


if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset()
    print(train_dataset[0])
    