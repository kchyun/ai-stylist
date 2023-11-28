import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from PIL.Image import Image


class FashionRecommender():
    def __init__(self, model, embed_generator, device):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.model = model
        self.model.eval()

        self.embed_generator = embed_generator

        self.model.to(self.device)
        self.embed_generator.to(self.device)

    @torch.no_grad()
    def single_infer(self, top, bottom, style):
        _, a_proj = self.model(torch.Tensor(top).to(self.device), style.to(self.device))
        _, b_proj = self.model(torch.Tensor(bottom).to(self.device), style.to(self.device))
        return torch.nn.PairwiseDistance(p=2)(a_proj, b_proj)
    