import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch


# Create dataloaders to iterate.
def CreateLoaders(train_dataset,val_dataset,config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False)

    return train_loader,val_loader
