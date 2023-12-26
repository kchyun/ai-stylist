import torch
import torch.nn as nn
from dataclasses import dataclass

from torch import Tensor
from typing import List

@ dataclass
class ModelArgs:
    src_embed_dim: int  = 512
    tgt_embed_dim: int = 64
    n_conditions: int=7

class StyleAwareNet(nn.Module):
    def __init__(
            self,
            args: ModelArgs,
            ):
        super(StyleAwareNet, self).__init__()
        self.src_embed_dim = args.src_embed_dim
        self.tgt_embed_dim = args.tgt_embed_dim

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.src_embed_dim, self.src_embed_dim),
            nn.BatchNorm1d(self.src_embed_dim),
            nn.Hardtanh(),
            nn.Linear(self.src_embed_dim, self.tgt_embed_dim),
            nn.Hardtanh()
            )
        
        # masks = []
        # for i in range(self.n_conditions):
        #     masks.append(nn.Linear(self.tgt_embed_size, self.tgt_embed_size))
        # self.masks = nn.ModuleList(masks)

        # self.masks = torch.nn.Embedding(self.n_conditions, args.tgt_embed_dim)
        # self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005


    def forward(self, x: Tensor):
        ''' x: Embedding of input images
        '''
        return [self.bottleneck_layer(x)] # CLIP 임베딩을 차원 축소한 것