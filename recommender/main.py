import argparse
import torch

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split

from model.fashion_mlp import FashionMLP
from utils.trainer import Trainer
from utils.data import get_dataset
from dataclasses import dataclass

@dataclass
class TrainingArgs:
    n_batch: int=256
    n_epochs: int=500
    learning_rate: float=0.0001
    device: str='cuda'

@dataclass
class ModelArgs:
    embed_size: int=512


def main():
    device = torch.device('cuda') if (TrainingArgs.device == 'cuda') & (torch.cuda.is_available()) else torch.device('cpu')

    train_dataset, test_dataset = get_dataset()
    train_dataset, valid_dataset = random_split(train_dataset, [0.9, 0.1])

    train_dataloader = DataLoader(train_dataset, TrainingArgs.n_batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, TrainingArgs.n_batch, shuffle=False)

    model = FashionMLP(ModelArgs).to(device)
    optimizer = Adam(model.parameters(), lr=TrainingArgs.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.9)
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer=optimizer, scheduler=scheduler, device=device, TrainingArgs=TrainingArgs)

    trainer.train()

    torch.save({'model' : trainer.best_model,
                'optimizer': trainer.best_optimizer})

if __name__ == '__main__':
    main()