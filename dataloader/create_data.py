import numpy as np
import pandas as pd
import random

def create_dataset(file_path):
    df = pd.read_csv(file_path,sep=',',index_col=None,usecols=['x','y','time','P','theta','phi'])
    df = df.to_numpy()
    random.shuffle(df)
    hits = df[:,:3]
    conds = df[:,3:]
    conds = (conds - conds.min(0)) / (conds.max(0) - conds.min(0))
    return hits,conds
