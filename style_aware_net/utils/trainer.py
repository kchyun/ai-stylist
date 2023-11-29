import os
import argparse
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, lr_scheduler
from model.loss import TripletLoss, compute_loss
from transformers import  CLIPVisionModelWithProjection, CLIPProcessor

from datetime import datetime
import wandb

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

        self.embed_model = CLIPVisionModelWithProjection.from_pretrained('patrickjohncyh/fashion-clip').to(device)
        for param in self.embed_model.parameters():
            param.requires_grad = False
        self.embed_model.eval()
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

    def train(self):
        
        
        lowest_loss = np.inf
        for epoch in range(self.args.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_loss = self._validate(self.valid_dataloader, epoch)
            if valid_loss <= lowest_loss:
               lowest_loss = valid_loss
               self.best_model_state = deepcopy(self.model.state_dict())
               self.best_optimizer_state = deepcopy(self.optimizer.state_dict())

            if epoch % self.args.save_every == 0:
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                model_name = f'{date}_{epoch}_{valid_loss:.3f}'
                self.save(self.args.save_path, model_name)
            


    def _train(self, dataloader, epoch):
        self.model.train()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            anc_img, pos_img, neg_random_imgs = batch

            anc_img = self.processor(images=anc_img, return_tensors="pt", padding=True)
            anc_embed = self.embed_model(**anc_img.to(self.device)).image_embeds.detach()
            anc_projs = self.model(anc_embed)

            pos_img = self.processor(images=pos_img, return_tensors="pt", padding=True)
            pos_embed = self.embed_model(**pos_img.to(self.device)).image_embeds.detach()
            pos_projs = self.model(pos_embed)

            # loss 가중치 결정
            style_logits = self.style_classifier.forward(anc_embed, pos_embed, self.device)
            loss_weight = style_logits 
            loss_weight = loss_weight.to(self.device)
            
            # 랜덤 샘플에 대한 Loss구하기
            B, N_S = neg_random_imgs.size(0), neg_random_imgs.size(1)
            neg_random_imgs = neg_random_imgs.view(-1, neg_random_imgs.size(2), neg_random_imgs.size(3), neg_random_imgs.size(4)) # (B * N_S, H, W, C)
            neg_random_imgs = self.processor(images=neg_random_imgs, return_tensors="pt", padding=True)
            neg_random_embeds = self.embed_model(**neg_random_imgs.to(self.device)).image_embeds.detach() 
            neg_random_projs = [
                neg_random_embed.view(B, N_S, -1)
                for neg_random_embed in self.model(neg_random_embeds)
                ]

            loss_random, logging_loss = compute_loss(anc_projs, pos_projs, neg_random_projs, loss_weight)

            # Augment sample로 만든 Loss와 Random sample로 만든 Loss를 가중치합
            loss = self.args.w_random * loss_random # + self.args.w_neg * loss_neg

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()

            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))
            
            wandb.log({"train_loss": loss_random,
                       "train_logging_loss": logging_loss}, commit=True)

        return losses / iter


    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.model.eval()
        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            anc_img, pos_img, neg_random_imgs = batch

            anc_img = self.processor(images=anc_img, return_tensors="pt", padding=True)
            anc_embed = self.embed_model(**anc_img.to(self.device)).image_embeds.detach()
            anc_projs = self.model(anc_embed)

            pos_img = self.processor(images=pos_img, return_tensors="pt", padding=True)
            pos_embed = self.embed_model(**pos_img.to(self.device)).image_embeds.detach()
            pos_projs = self.model(pos_embed)

            # loss 가중치 결정
            style_logits = self.style_classifier.forward(anc_embed, pos_embed, self.device)
            loss_weight = style_logits 
            loss_weight = loss_weight.to(self.device)
            
            # 랜덤 샘플에 대한 Loss구하기
            B, N_S = neg_random_imgs.size(0), neg_random_imgs.size(1)
            neg_random_imgs = neg_random_imgs.view(-1, neg_random_imgs.size(2), neg_random_imgs.size(3), neg_random_imgs.size(4)) # (B * N_S, H, W, C)
            neg_random_imgs = self.processor(images=neg_random_imgs, return_tensors="pt", padding=True)
            neg_random_embeds = self.embed_model(**neg_random_imgs.to(self.device)).image_embeds.detach() 
            neg_random_projs = [
                neg_random_embed.view(B, N_S, -1)
                for neg_random_embed in self.model(neg_random_embeds)
                ]

            loss_random, logging_loss = compute_loss(anc_projs, pos_projs, neg_random_projs, loss_weight)

            # Augment sample로 만든 Loss와 Random sample로 만든 Loss를 가중치합
            loss = self.args.w_random * loss_random # + self.args.w_neg * loss_neg
            
            losses += loss.item()

            epoch_iterator.set_description(
                'Valid | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, losses / iter))
            
            wandb.log({"valid_loss": loss_random,
                       "valid_logging_loss": logging_loss}, commit=True)

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