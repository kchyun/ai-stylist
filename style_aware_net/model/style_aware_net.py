import torch
import torch.nn as nn
from dataclasses import dataclass

from torch import Tensor
from typing import List

@ dataclass
class ModelArgs:
    src_embed_dim: int  = 512
    tgt_embed_dim: int = 32
    n_conditions: int=7

class StyleAwareNet(nn.Module):
    def __init__(
            self,
            args: ModelArgs,
            ):
        super(StyleAwareNet, self).__init__()
        self.src_embed_dim = args.src_embed_dim
        self.tgt_embed_dim = args.tgt_embed_dim
        self.n_conditions = args.n_conditions

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.src_embed_dim, self.src_embed_dim),
            nn.BatchNorm1d(self.src_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.src_embed_dim, self.src_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.src_embed_dim, self.tgt_embed_dim),
            )

        self.masks = torch.nn.Embedding(self.n_conditions, args.tgt_embed_dim)
        self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
    
    
    def forward(self, x: Tensor, s: Tensor=None):
        comp_embed = self.bottleneck_layer(x)

        if s is None:
            proj_embeds = []
            for i in range(self.n_conditions):
                mask = self.masks(torch.LongTensor([i]).cuda())
                proj_embed = nn.LeakyReLU()(comp_embed) * mask
                proj_embeds.append(proj_embed)
            return proj_embeds
        else:
            mask = self.masks(s)
            proj_embed = nn.LeakyReLU()(comp_embed) * mask
            return proj_embed