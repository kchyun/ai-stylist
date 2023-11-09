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
    def single_infer(self, style_dict: Dict[str, Image]):
        embeds = [self.embed_generator.img2embed(style_dict[key]) for key in style_dict.keys()]
        logits = self.model(*embeds)
        return logits