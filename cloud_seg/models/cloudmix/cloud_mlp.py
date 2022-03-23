"""
Train MLP to match cloudless versions of images to cloudy ones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from torch.optim import Adam

import numpy as np

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class LitMLP(pl.LightningModule):

    def __init__(self, data, labels, val_frac=0.25, batch_size=64, learning_rate=1e-1):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.num_bands = data.shape[1]
        self.learning_rate = learning_rate            
        self.batch_size=batch_size
       
        
        self.fc1 = nn.Linear(self.num_bands, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, self.num_bands)

        # self.fc1 = nn.Linear(self.num_bands, 8)
        # self.fc2 = nn.Linear(8, self.num_bands)

        # self.fc1 = nn.Linear(self.num_bands, self.num_bands)

        nsamples = data.shape[0]
        inds = np.arange(nsamples)
        np.random.shuffle(inds)
        
        ind_split = int(nsamples*(1-val_frac))
        self.train_data = data[inds[:ind_split]]
        self.train_labels = labels[inds[:ind_split]]
        
        self.val_data = data[inds[ind_split:]]
        self.val_labels = labels[inds[ind_split:]]

        self.train_dataset = MLPDataset(self.train_data, self.train_labels)
        self.val_dataset = MLPDataset(self.val_data, self.val_labels)

        # self.criteria = nn.L1Loss()
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x = self.fc1( x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        # loss = self.criteria(preds, y)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss)
        return loss
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #         optimizer,
        #         gamma=0.99,
        #     )
        return optimizer
        # return [optimizer], [scheduler]

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
    
    def val_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        ) 
    
class MLPDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, data, labels):
    
        self.data  = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        return self.data[idx], self.labels[idx]

