import argparse
import torch

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split

from model.fashion_mlp import FashionMLP
from utils.trainer import Trainer
from utils.data import get_dataset

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)

    p.add_argument('--use_linear', type=bool, default=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=0.0001)

    p.add_argument('--n_epochs', type=int, default=40)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--embed_size', type=int, default=512)

    config = p.parse_args()
    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_dataset, test_dataset = get_dataset()
    train_dataset, valid_dataset = random_split(train_dataset, [0.9, 0.1])

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, config.batch_size // 4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, config.batch_size // 4, shuffle=False)

    model = FashionMLP(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size = 100,
            gamma=0.9
        )
    
    trainer = Trainer(model, train_dataloader, valid_dataloader, optimizer=optimizer, scheduler=scheduler, config=config)

    trainer.train()

    torch.save({
        'model' : trainer.best_model,
        'opt': trainer.best_optimizer,
        'config': config
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    
    main(config)