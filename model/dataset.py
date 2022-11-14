import torch

def custom_collate(data):
    features = torch.stack([torch.tensor([d['feature']]) for d in data])
    labels = torch.tensor([d['label'] for d in data])
    new_batch = {
        'feature': features,
        'label': labels
    }
    return new_batch
