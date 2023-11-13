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

PAIRS_PATH = "C:/Users/owj04/Desktop/Projects/ai-stylist/datasets/FashionVCdata/FashionVCdata/gnd_top_bottom_pairs.csv"
TOP_EMBEDS_PATH = "C:/Users/owj04/Desktop/Projects/ai-stylist/datasets/FashionVCdata/FashionVCdata/top_embeds.json"
BOTTOM_EMBEDS_PATH = "C:/Users/owj04/Desktop/Projects/ai-stylist/datasets/FashionVCdata/FashionVCdata/bottom_embeds.json"

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
            neg_bottom_idxs = self._get_neg_bottom(top_idx, n=128)

            source_embed = self.top_embeds[top_idx]
            pos_embed = self.bottom_embeds[bottom_idx]
            neg_embeds = [self.bottom_embeds[neg_bottom_idx] for neg_bottom_idx in neg_bottom_idxs]

            item = (source_embed, pos_embed, neg_embeds)
        else:
            top_idx, bottom_idx = self.data[idx]
            item = {'input_embed' : self.top_embeds[top_idx],
                    'target' : bottom_idx}
            
        return item
            
    def _get_neg_bottom(self, top_idx, n):
        neg_bottom_idxs = []
        while len(neg_bottom_idxs) < 128:
            neg_bottom_idx = self.data[random.randint(0, len(self.data) - 1)][-1]
            if neg_bottom_idx not in self.pairs[top_idx]:
                neg_bottom_idxs.append(str(neg_bottom_idx))
        return neg_bottom_idxs
        
    def _preprocess_embeds(self, path):
        with open(path, 'r') as f:
            items = json.load(f)
        return {str(item['id']): Tensor(item['embed']) for item in items}

class FashionSIAMESEDataset(Dataset):
    def __init__(
        self,
        pairs: Dict[str, Set[str]],
        dataset_type: str='train'
        ):
        super(FashionSIAMESEDataset).__init__()
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

            exclude_indices = [top_idx, bottom_idx, neg_bottom_idx]

            source_embed = self.top_embeds[top_idx]
            pos_embed = self.bottom_embeds[bottom_idx]
            neg_embed = self.bottom_embeds[neg_bottom_idx]

            # Sample a subset excluding the current embeddings, to find similar embeddings
            top_sampled_embeds, bottom_sampled_embeds = self._sample_subset(exclude_indices)

            # Find the most similar embeddings within the subset
            source_sim_idx = self._find_most_similar(source_embed, top_sampled_embeds)
            pos_sim_idx = self._find_most_similar(pos_embed, bottom_sampled_embeds)
            neg_sim_idx = self._find_most_similar(neg_embed, bottom_sampled_embeds)

            source_sim_embed = self.top_embeds.get(source_sim_idx, self.top_embeds[source_sim_idx])
            pos_sim_embed = self.bottom_embeds.get(pos_sim_idx, self.bottom_embeds[pos_sim_idx])
            neg_sim_embed = self.bottom_embeds.get(neg_sim_idx, self.bottom_embeds[neg_sim_idx])

            item = (source_embed, pos_embed, neg_embed, source_sim_embed, pos_sim_embed, neg_sim_embed)
        else:
            top_idx, bottom_idx = self.data[idx]
            item = {'input_embed': self.top_embeds[top_idx], 'target': bottom_idx}

        return item

    def _get_neg_bottom(self, top_idx):
        while True:
            neg_bottom_idx = self.data[random.randint(0, len(self.data) - 1)][-1]
            if neg_bottom_idx not in self.pairs[top_idx]:
              return str(neg_bottom_idx)

    def _sample_subset(self, exclude_indices, num_samples=512):
        top_keys = set(self.top_embeds.keys())
        bottom_keys = set(self.bottom_embeds.keys())
        exclude_indices = [str(idx) for idx in exclude_indices]

        top_available_indices = list(top_keys - set([exclude_indices[0]]))
        bottom_available_indices = list(bottom_keys - set(exclude_indices[1:]))
        top_sampled = random.sample(top_available_indices, min(num_samples, len(top_available_indices)))
        bottom_sampled = random.sample(bottom_available_indices, min(num_samples, len(bottom_available_indices)))
        # key type : str
        top_sampled_embeds = {idx: self.top_embeds[idx] for idx in top_sampled}
        bottom_sampled_embeds = {idx: self.bottom_embeds[idx] for idx in bottom_sampled}

        return top_sampled_embeds, bottom_sampled_embeds

    def _find_most_similar(self, embed, subset_embeds):
        similarities = {idx: torch.nn.functional.cosine_similarity(embed, other_embed, dim=0) for idx, other_embed in subset_embeds.items() if other_embed is not None}

        if not similarities:
            raise ValueError("No similarities found in the subset embeddings.")

        most_similar_idx = max(similarities, key=similarities.get)
        return most_similar_idx

    def _preprocess_embeds(self, path):
        with open(path, 'r') as f:
            items = json.load(f)
        return {str(item['id']): Tensor(item['embed']) for item in items}


def get_dataset(model_type='MLP') -> Tuple[FashionMLPDataset, FashionMLPDataset]:
    print('Dataset generating starts...')
    pair_df = pd.read_csv(PAIRS_PATH)
    pairs = defaultdict(list)
    for i, (top, bottom) in pair_df.iterrows():
        pairs[str(top)].append(str(bottom))
    train_pairs, test_pairs = _hit_rate_split(pairs)

    if model_type == 'MLP':
        return (FashionMLPDataset(train_pairs, 'train'),
                FashionMLPDataset(test_pairs, 'test'))
    elif model_type == 'SIAMESE':
        return (FashionSIAMESEDataset(train_pairs, 'train'),
                FashionSIAMESEDataset(test_pairs, 'test'))
    
    
def _hit_rate_split(pairs: Dict[str, List[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    print('Train, test spliting starts...')
    train_pairs, test_pairs = dict(), dict()
    for top_idx, bottom_list in pairs.items():
        test_bottom = bottom_list.pop()
        train_pairs[top_idx] = set(bottom_list)
        test_pairs[top_idx] = set(test_bottom)
    return train_pairs, test_pairs
    