import os
import argparse
import torch

from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split

import albumentations as A

from model.style_aware_net import *
from model.style_classifier import *
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from embed_generator.generator import *
from utils.trainer import Trainer
from utils.data import get_dataset
from dataclasses import dataclass

import wandb

LOAD_PATH = 0

MODEL_NAME = 'Final'

@dataclass
class TrainingArgs:
    n_batch: int=32
    n_epochs: int=10
    learning_rate: float=0.001
    device: str='cuda'
    save_every: int=1
    save_path: str='F:/Projects/2023-ai-stylist/style_aware_net/model/saved_model'
    w_random: int=1


model_args = ModelArgs(
    n_conditions = 6
)

styles = [
       "formal business meeting",
       "athletic running",
       "casual day", 
       "hip-hop", 
       "party", 
       "beach vacation",
       ]


def main():
    os.environ["WANDB_API_KEY"] = "fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f"
    os.environ["WANDB_PROJECT"] = "ai-stylist"
    os.environ["WANDB_LOG_MODEL"] = "all"

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="sytle-7"
        )
    


    device = torch.device('cuda') if (TrainingArgs.device == 'cuda') & (torch.cuda.is_available()) else torch.device('cpu')

    transform = A.Compose([
        A.HorizontalFlip(),
        A.RandomResizedCrop(scale=(0.8, 1.2), height=300, width=300, p=0.5),
        A.Rotate(limit=15),
        
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0),
        A.GaussNoise(),
        ])

    train_dataset, valid_dataset = get_dataset(transform)

    train_dataloader = DataLoader(train_dataset, TrainingArgs.n_batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, TrainingArgs.n_batch, shuffle=False)

    model = StyleAwareNet(model_args).to(device)

    # MODEL_PATH = 'F:/Projects/2023-ai-stylist/style_aware_net/model/saved_model'
    # MODEL_NAME = '2023-11-29_0_1.515'   
    # model_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')
    # model.load_state_dict(torch.load(model_path))
    # print(f'Model successfully loaded')


    optimizer = AdamW(model.parameters(), lr=TrainingArgs.learning_rate, weight_decay=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)

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