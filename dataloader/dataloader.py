import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch


def DIRC_collate(batch):
    hits = []
    conditions = []
    PIDs = []
    metadata = []
    unscaled = []
    for h,cond,PID,meta,u in batch:
        hits.append(torch.tensor(h))
        conditions.append(torch.tensor(cond))
        PIDs.append(torch.tensor(PID))
        metadata.append(torch.tensor(meta))
        unscaled.append(torch.tensor(u))

    return torch.stack(hits),torch.stack(conditions),torch.tensor(PIDs),torch.stack(metadata),torch.stack(unscaled)

# Create dataloaders to iterate.
def CreateLoaders(train_dataset,val_dataset,config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,collate_fn=DIRC_collate,num_workers=8)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate,num_workers=8)

    return train_loader,val_loader
