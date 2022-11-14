import multiprocessing
import datasets
import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.dataset import custom_collate


class CNN(LightningModule):
    def __init__(self, c1_output_dim, c1_kernel_size, c1_strides, c2_output_dim, c2_kernel_size, c2_strides,
                 output_dim, data_path, signal_length=1500):
        super().__init__()
        self.save_hyperparameters()

        self.c1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_strides
            ),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_strides
            ),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool1d(
            kernel_size=2
        )

        #calculate the output size of max pool     
        x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        max_pool_out = x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=max_pool_out,
                out_features=200
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=200,
                out_features=100
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(
            in_features=50,
            out_features=self.hparams.output_dim
        )

    def forward(self, x):
    
        inp_dim = x.shape[0]

        x = self.c1(x)
        x = self.c2(x)
        x = self.max_pool(x)

        x = x.reshape(inp_dim, -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return  self.out(x)

    def train_dataloader(self):
        # expect to get train folder
        dataset_dict = datasets.load_dataset(self.hparams.data_path)
        dataset = dataset_dict[list(dataset_dict.keys())[0]]
        return DataLoader(dataset, batch_size=16, collate_fn=custom_collate, shuffle=True)

    def training_step(self, batch, batch_idx):
        x = batch['feature'].float()
        y = batch['label'].long()
        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer