import os
import argparse
import torch

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split

from model.style_aware_net import *
from model.style_classifier import *
from utils.trainer import Trainer
from utils.data import get_dataset
from dataclasses import dataclass

LOAD_PATH = 0
MODEL_PATH = './'
MODEL_NAME = 'tmp'

@dataclass
class TrainingArgs:
    n_batch: int=256
    n_epochs: int=18
    learning_rate: float=0.0001
    device: str='cuda'


model_args = ModelArgs(
    n_conditions = 10
)

styles = ["formal and modern", "sports", "casual", "ethnic and hippie", "hip-hop", "preppy", 'feminine']


def main():
    device = torch.device('cuda') if (TrainingArgs.device == 'cuda') & (torch.cuda.is_available()) else torch.device('cpu')

    train_dataset, valid_dataset, test_dataset = get_dataset()
    # train_dataset, valid_dataset = random_split(train_dataset, [0.9, 0.1])

    train_dataloader = DataLoader(train_dataset, TrainingArgs.n_batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, TrainingArgs.n_batch, shuffle=False)

    model = StyleAwareNet(model_args).to(device)
    optimizer = Adam(model.parameters(), lr=TrainingArgs.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.9)

    style_classifier = StyleClassifier(styles=styles)
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer=optimizer, scheduler=scheduler, style_classifier=style_classifier, device=device, args=TrainingArgs)

    trainer.train()
    trainer.save(MODEL_PATH, MODEL_NAME) 


if __name__ == '__main__':
    main()