import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from loss import PairwiseRankingLoss


class Trainer:

    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, device, TrainingArgs):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.device = device
        self.TrainingArgs = TrainingArgs

        self.best_model = None
        self.best_optimizer = None

    def train(self):
        lowest_loss = np.inf

        for epoch in range(self.TrainingArgs.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_loss = self._validate(self.valid_dataloader, epoch)
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                self.best_optimizer_state = deepcopy(self.optimizer.state_dict())
            

    def _train(self, dataloader, epoch):
        self.model.train()

        epoch_iterator = tqdm(dataloader)

        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            source_embed, pos_embed, neg_embeds = batch #train_loader가 어떻게 되냐에 따라 이부분도 바뀔 것 같습니다.


            _, pos =  self.model([source_embed, pos_embed])
            neg = torch.mean(torch.stack([self.model([source_embed.to(self.device), neg_embed.to(self.device)])[1] for neg_embed in neg_embeds]), dim=0)

            loss = PairwiseRankingLoss(pos, neg)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses += loss.item()
            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )
        
        return losses / iter
    
    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            source_embed, pos_embed, neg_embeds = batch
            _, pos =  self.model([source_embed, pos_embed])
            neg = torch.mean(torch.stack([self.model([source_embed.to(self.device), neg_embed.to(self.device)])[1] for neg_embed in neg_embeds]), dim=0)

            loss = PairwiseRankingLoss(pos, neg)
            
            losses += loss.item()
            epoch_iterator.set_description(
                'Validation | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )
        return losses / iter

    def save(self, model_path, model_name, best_model: bool=True):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        if best_model:
            torch.save(self.best_model_state, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        print(f'Model successfully saved at {model_path}')


    def load(self, model_path, model_name):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        self.model.load_state_dict(torch.load(model_path))
        print(f'Model successfully loaded from {model_path}')


class Mlm_Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, loss, optimizer, scheduler, device, TrainingArgs):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.mlm_loss = loss
        self.scheduler = scheduler
        self.device = device
        self.TrainingArgs = TrainingArgs

        self.best_model = None
        self.best_optimizer = None

    def train(self):
        lowest_loss = np.inf

        for epoch in range(self.TrainingArgs.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_loss = self._validate(self.valid_dataloader, epoch)
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                self.best_optimizer_state = deepcopy(self.optimizer.state_dict())
            

    def _train(self, dataloader, epoch):
        self.model.train()

        epoch_iterator = tqdm(dataloader)

        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            source_embed, pos_embed, context_embeds = batch #아마 mlm train_loader는 다음처럼 3개를 가져올 것 같습니다.
            

            labels, output =  self.model([source_embed, pos_embed], context_embeds, True) #embedding을 모두 묶어서 보내고, context_embed는 따로 보냅니다.
            
            loss = self.loss(output, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses += loss.item()
            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )
        
        return losses / iter
    
    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            source_embed, pos_embed, context_embeds = batch #아마 mlm train_loader는 다음처럼 3개를 가져올 것 같습니다.
            
            labels, output =  self.model([source_embed, pos_embed], context_embeds, True) #embedding을 모두 묶어서 보내고, context_embed는 따로 보냅니다.
            
            loss = self.loss(output, labels)
            
            losses += loss.item()
            epoch_iterator.set_description(
                'Validation | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.TrainingArgs.n_epochs, losses / iter)
                )
        return losses / iter

    def save(self, model_path, model_name, best_model: bool=True):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        if best_model:
            torch.save(self.best_model_state, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        print(f'Model successfully saved at {model_path}')


    def load(self, model_path, model_name):
        model_path = os.path.join(model_path, f'{model_name}.pth')
        self.model.load_state_dict(torch.load(model_path))
        print(f'Model successfully loaded from {model_path}')