import json
import random
from collections import OrderedDict, defaultdict

import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Optional, Dict, List, Set, Tuple

from copy import deepcopy

TRAIN_PAIRS_PATH = 'F:/Projects/ai-stylist/data/polyvore_cleaned/train.json'
VALID_PAIRS_PATH = 'F:/Projects/ai-stylist/data/polyvore_cleaned/valid.json'
TEST_PAIRS_PATH = 'F:/Projects/ai-stylist/data/polyvore_cleaned/test.json'

TOP_EMBEDS_PATH = 'F:/Projects/ai-stylist/data/polyvore_cleaned/top_embeds.json'
BOTTOM_EMBEDS_PATH = 'F:/Projects/ai-stylist/data/polyvore_cleaned/bottom_embeds.json'

class StyleAwareNetDataset(Dataset):
    def __init__(
        self,
        path,
        ):
        super(StyleAwareNetDataset, self).__init__()
        self.pos_df, self.neg_df = self._preprocess_pairs(path)
        self.anc_ids = self.pos_df.index
        self.top_embeds = self._preprocess_embeds(TOP_EMBEDS_PATH)
        self.bottom_embeds = self._preprocess_embeds(BOTTOM_EMBEDS_PATH)
        
        
    def __len__(self):
        return len(self.ancs)
    

    def __getitem__(self, idx):
        anc_id = self.anc_ids[idx]

        pos_id = random.choice(self.pos_df.loc[anc_id]['bottom_id'])
        neg_id = random.choice(self.neg_df.loc[anc_id]['bottom_id'])

        anc = self.top_embeds.loc[anc_id]['embed']
        pos = self.bottom_embeds.loc[pos_id]['embed']
        neg = self.bottom_embeds.loc[neg_id]['embed']
        return anc, pos, neg
    

    def _preprocess_pairs(self, path):
        df = pd.read_json(path).groupby(['top_id', 'y']).agg({'bottom_id' : list})
        pos_df = df.xs(1, level='y')
        neg_df = df.xs(0, level='y')
        return pos_df, neg_df
        

    def _preprocess_embeds(self, path):
        return pd.read_json(path)
    

def get_dataset():
    train_dataset = StyleAwareNetDataset(TRAIN_PAIRS_PATH)
    valid_dataset = StyleAwareNetDataset(VALID_PAIRS_PATH)
    test_dataset = StyleAwareNetDataset(TEST_PAIRS_PATH)
    return train_dataset, valid_dataset, test_dataset

if __name__ == '__main__':
    t, v, test = get_dataset()