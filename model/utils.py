import multiprocessing
from pathlib import Path
import datasets
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader


def dataset_collate(batch):
    feature = torch.stack([torch.tensor([data['feature']]) for data in batch])
    label = torch.tensor([data['label'] for data in batch])
    transformed_batch = {
        'feature': feature,
        'label': label
    }
    return 

def confusion_matrix(data_path, model, num_class):
    
    model.eval()
    matrix = np.zeros((num_class, num_class), dtype=np.float)
    dataset_dict = datasets.load_dataset(data_path)
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    
    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1
    
    dataloader = DataLoader(dataset, batch_size=4096, num_workers=num_workers, collate_fn=dataset_collate)
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