import sys
sys.path.append('C:/KU/ai-stylist/ai-stylist')

from torchmetrics.functional import pairwise_cosine_similarity
from embed_generator.generator import *
import torch

class StyleClassifier():
    def __init__(self, 
                 styles: List[str]):
        
        self.styles = styles
        self.model = FashionEmbeddingGenerator()
        self.prompt_embeddings = torch.stack([torch.Tensor(self.model.text2embed("a photo of {} style clothes".format(s))) for s in styles])
        
    def forward(self, anc, pos):
        
        # import pdb; pdb.set_trace()
        
        pos_similarity = pairwise_cosine_similarity(pos, self.prompt_embeddings.squeeze(1))
        anc_similarity = pairwise_cosine_similarity(anc, self.prompt_embeddings.squeeze(1))
        
        anc_similarity = torch.nn.functional.softmax(anc_similarity, dim=1)
        pos_similarity = torch.nn.functional.softmax(pos_similarity, dim=1)
        
        return torch.argmax(torch.add(anc_similarity, pos_similarity) / 2, dim=1)

    def idx2stl(self, idx):
        return self.styles[idx]
        
    def stl2idx(self, stl):
        return self.styles.index(stl)