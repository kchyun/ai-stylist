import os
import argparse
import torch

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split

import albumentations as A

from model.style_aware_net import *
from model.style_classifier import *
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from embed_generator.generator import *
from utils.trainer import Trainer
from utils.data import get_dataset
from dataclasses import dataclass

LOAD_PATH = 0

MODEL_NAME = 'Final'

@dataclass
class TrainingArgs:
    n_batch: int=32
    n_epochs: int=5
    learning_rate: float=0.0001
    device: str='cuda'
    save_every: int=1
    save_path: str = 'C:/KU/ai-stylist/ai-stylist/style_aware_net/model/saved_model'


model_args = ModelArgs(
    n_conditions = 7
)

# styles = ["formal, dandy and minimal",
#           "athletic and sports",
#           "casual and classic", 
#           "ethnic, hippie and maximalism", 
#           "hip-hop, street and gangster", 
#           "preppy and classic", 
#           "feminine and girlish"]

styles = [
    "wedding",
    "casual date",
    "party",
    "workout and sports",
    "funeral",
    "trip",
    "work and business",
    "formal meeting",
    "meeting friends",
]



def main():
    device = torch.device('cuda') if (TrainingArgs.device == 'cuda') & (torch.cuda.is_available()) else torch.device('cpu')

    transform = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(limit=15),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ])
    
    train_dataset, valid_dataset = get_dataset(transform)

    train_dataloader = DataLoader(train_dataset, TrainingArgs.n_batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, TrainingArgs.n_batch, shuffle=False)

    model = StyleAwareNet(model_args).to(device)
    optimizer = Adam(model.parameters(), lr=TrainingArgs.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.9)

    embed_generator = FashionEmbeddingGenerator()
    style_classifier = StyleClassifier(embed_generator=embed_generator, styles=styles)
    
    args = TrainingArgs()
    trainer = Trainer(
        model,
        train_dataloader,
        valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        style_classifier=style_classifier,
        device=device,
        args=args)

    trainer.train()
    trainer.save(args.save_path, MODEL_NAME) 


if __name__ == '__main__':
    main()