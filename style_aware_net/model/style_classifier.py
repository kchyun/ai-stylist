import sys
sys.path.append('c:/KU/ai-stylist/ai-stylist/')

from embed_generator.generator import *
import torch

class StyleClassifier():
    def __init__(self, 
                 styles: List[str]):
        
        self.styles = styles
        self.model = FashionEmbeddingGenerator()
        self.prompt_embeddings = torch.stack([torch.Tensor(self.model.text2embed("a photo of {} style clothes".format(s))) for s in styles])
        
    def forward(self, anc, pos):
        anc_similarity = torch.stack([torch.nn.CosineSimilarity(anc, e) for e in self.prompt_embeddings])
        pos_similarity = torch.stack([torch.nn.CosineSimilarity(pos, e) for e in self.prompt_embeddings])
        
        anc_similarity = torch.nn.functional.softmax(anc_similarity)
        pos_similarity = torch.nn.functional.softmax(pos_similarity)
        
        return torch.argmax(torch.add(anc_similarity, pos_similarity) / 2)

    def idx2stl(self, idx):
        return self.styles[idx]
        
    def stl2idx(self, stl):
        return self.styles.index(stl)