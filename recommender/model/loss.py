import torch
from torch import Tensor

def PairwiseRankingLoss(i: Tensor, j: Tensor):
    return - (i - j).sigmoid().log().mean()