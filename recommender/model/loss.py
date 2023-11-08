import torch
from torch import Tensor
from torch.nn import TripletMarginLoss

def PairwiseRankingLoss(pos, neg):
    return torch.log(1 + torch.exp(-(pos - neg))).mean()

def TripletLoss(pos_top_embed, pos_bottom_embed, neg_bottom_embed):
    return TripletMarginLoss()(pos_top_embed, pos_bottom_embed, neg_bottom_embed)