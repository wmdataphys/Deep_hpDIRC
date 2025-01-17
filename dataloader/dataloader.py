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


def Inference_collate(batch):
    hits = []
    conditions = []
    PIDs = []
    unscaled = []
    n_hits = []
    LL_ks = []
    LL_pis = []
    for h,cond,PID,nh,u,LL_k,LL_pi in batch:
        hits.append(torch.tensor(h))
        conditions.append(torch.tensor(cond))
        PIDs.append(torch.tensor(PID))
        n_hits.append(torch.tensor(nh))
        unscaled.append(torch.tensor(u))
        LL_ks.append(torch.tensor(LL_k))
        LL_pis.append(torch.tensor(LL_pi))

    return torch.stack(hits),torch.stack(conditions),torch.tensor(PIDs),torch.tensor(n_hits),torch.stack(unscaled),torch.tensor(LL_ks),torch.tensor(LL_pis)

# Create dataloaders to iterate.
# Remember if training on linux you can use multiple workers.
def CreateLoaders(train_dataset,val_dataset,config,model_type=None):
    if model_type is None:
        raise ValueError("Model Type is None. Specify.")

    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size_'+str(model_type)],
                            shuffle=True,collate_fn=DIRC_collate,num_workers=0)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size_'+str(model_type)],
                            shuffle=False,collate_fn=DIRC_collate,num_workers=0)

    return train_loader,val_loader

# Create dataloaders to iterate.
def CreateInferenceLoader(test_dataset,config):
    test_loader =  DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size'],
                            shuffle=False,collate_fn=Inference_collate,num_workers=0)
    return test_loader
