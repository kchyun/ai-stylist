import os
import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from model.loss import TripletLoss


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, scheduler, style_classifier, device, args):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.device = device
        self.args = args

        self.style_classifier = style_classifier

        self.best_model_state = None
        self.best_optimizer_state = None

    def train(self):
        lowest_loss = np.inf
        for epoch in range(self.args.n_epochs):
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

            anc, pos, neg = batch
            style_type = self.style_classifier.forward(anc, pos).to(self.device)

            _, anc_proj = self.model(anc.to(self.device), style_type)
            _, pos_proj = self.model(pos.to(self.device), style_type)
            _, neg_proj = self.model(neg.to(self.device), style_type)

            loss = TripletLoss(anc_proj, pos_proj, neg_proj)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()

            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))

        return losses / iter


    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            anc, pos, neg = batch
            style_type = self.style_classifier.forward(anc, pos)

            anc_proj = self.model(anc.to(self.device), style_type)
            pos_proj = self.model(pos.to(self.device), style_type)
            neg_proj = self.model(neg.to(self.device), style_type)

            loss = TripletLoss(anc_proj, pos_proj, neg_proj)
            
            losses += loss.item()

            epoch_iterator.set_description(
                'Valid | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))

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