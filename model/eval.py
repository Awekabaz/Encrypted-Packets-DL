from pathlib import Path
import datasets
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.dataset import custom_collate
from model.CNN_model import CNN

def confusion_matrix(data_path, model, class_num):
    
    model.eval()
    matrix = np.zeros((class_num, class_num), dtype=np.float)
    
    dataset_dict = datasets.load_dataset(data_path)
    dataset = dataset_dict[list(dataset_dict.keys())[0]]     
    dataloader = DataLoader(dataset, batch_size=4096, collate_fn=custom_collate)
    
    for batch in dataloader:
        x = batch['feature'].float().to(model.device)
        y = batch['label'].long()
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)

        for i in range(len(y)):
            matrix[y[i], y_hat[i]] += 1

    return matrix

def get_classification_report(matrix, labels=None):
    
    if labels is None:
        labels = [i for i in range(matrix.shape[0])]   
    
    report = [{'label': labels[i], 
             'precision': matrix[i, i] / matrix[:, i].sum(), 
             'recall': matrix[i, i] / matrix[i, :].sum()} for i in range(matrix.shape[0])]

    return pd.DataFrame(report)


def load_model(model_path, gpu):
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    #model = CNN.load_from_checkpoint(str(Path(model_path).absolute()), map_location=torch.device(device)).float().to(device)
    model = CNN.load_from_checkpoint(model_path, map_location=torch.device(device)).float().to(device)
    model.eval()

    return model