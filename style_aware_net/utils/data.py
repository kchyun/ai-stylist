import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        neg_ids = self._get_neg_ids(anc_id, pos_id, n_sample=8, n_candidates=1024, threshold=0.7)

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
    
    def get_embedding(self, img_id):
        # function to get embeddings for cosine similarity calculation - 에러 나는 부분
        return 0

    def _get_neg_ids(self, anc_id, pos_id, n_sample=8, n_candidates = 1024, threshold = 0.7):
        # random sample, ensuring not to include positive sample
        candidates = random.sample([id for id in self.bottom_ids if id!=pos_id], n_candidates)
        pos_embedding = self._get_embedding(pos_id)
        candidate_embeddings = torch.stack([self._get_embedding(candidate) for candidate in candidates])

        similarities = F.cosine_similarity(pos_embedding.unsqueeze(0), candidate_embeddings)

        # Sort by similarity in descending order, similarity below threshold ensures that filtered candidates are at least "negative samples".
        filtered_candidates = [(candidate, similarity.item()) for candidate, similarity in zip(candidates, similarities) if similarity < threshold]
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select the top `n_sample` negatives
        neg_ids = [candidate for candidate, _ in filtered_candidates[:n_sample]]
        return neg_ids

    """
    1. Asusming we have bottom embedding dataset. Using EM algorithm, cluster them into n clusters. 
    n will be the argument.
    2. Positive embedding (embedding vector corresponding to pos_id) will be in certain cluster A. 
    We will sample n_sample negatives in different cluster with that positive embedding.
    3. Each cluster will have mean vector. We will sample negatives from cluster that has centroid (mean vector) 
    among n-1 clusters that are closest from centroid of A. Such cluster will be B.
    4. In B cluster, randomly sample n_sample negatives.
    """
    def _get_neg_ids_cluster(self, pos_id, n_sample=8):
        pos_cluster_id = self.clusters[pos_id]
        pos_centroid = self.centroids[pos_cluster_id]

        # Calculate distances of centroids to the positive centroid and sort them
        distances = torch.cdist([pos_centroid], self.centroids, p=2).flatten()
        closest_clusters = np.argsort(distances)

        # Find the closest different cluster
        for cluster_id in closest_clusters:
            if cluster_id != pos_cluster_id:
                closest_diff_cluster_id = cluster_id
                break

        # Get bottom ids in the closest different cluster
        closest_diff_cluster_bottoms = [img_id for img_id, cluster_id in self.clusters.items() if cluster_id == closest_diff_cluster_id]

        # Randomly sample negatives from this cluster
        neg_ids = random.sample(closest_diff_cluster_bottoms, min(n_sample, len(closest_diff_cluster_bottoms)))

        return neg_ids
    

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