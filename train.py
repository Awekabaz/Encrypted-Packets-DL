from pathlib import Path
import os
import click
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from model.CNN import CNN


def train_cnn(c1_kernel_size, c1_output_dim, c1_stride, c2_kernel_size, c2_output_dim, c2_stride, output_dim, data_path,
              epoch, model_path, signal_length, logger):

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed=1337, workers=True)

    CNN_model = CNN(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c2_kernel_size=c2_kernel_size,
        c2_output_dim=c2_output_dim,
        c2_stride=c2_stride,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()

    trainer = Trainer(val_check_interval=1.0, max_epochs=epoch, devices='auto', accelerator='auto', logger=logger,
                      callbacks=[EarlyStopping(monitor='training_loss', mode='min', check_on_train_epoch_end=True)])
    trainer.fit(CNN_model)

    # save model
    trainer.save_checkpoint(model_path)


@click.command()
@click.option('--data', help='a path to folder containing training data', required=True)
@click.option('--model', help='output model path', required=True)
@click.option('--task', help='Task type. Options: "app" or "traffic"', required=True)

def main(data, model, task):
    if task == 'app':
        # train_application_cnn
        app_logger = TensorBoardLogger('app_CNN_logs', 'app_CNN')
        train_cnn(c1_kernel_size=4, c1_output_dim=200, c1_stride=3, c2_kernel_size=5, c2_output_dim=200, c2_stride=1,
                output_dim=17, data_path=data, epoch=20, model_path=model, signal_length=1500, logger=app_logger)
    
    elif task == 'traffic':
        #train_traffic_cnn
        traffic_logger = TensorBoardLogger('traffic_CNN_logs', 'traffic_CNN')
        train_cnn(c1_kernel_size=5, c1_output_dim=200, c1_stride=3, c2_kernel_size=4, c2_output_dim=200, c2_stride=3,
                output_dim=12, data_path=data, epoch=20, model_path=model, signal_length=1500, logger=traffic_logger)
 
    else:
        exit('Given Task is not supported')

if __name__ == '__main__':
    main()
    
