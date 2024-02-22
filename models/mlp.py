import torch
from torch import nn
from torch.nn import BatchNorm1d

class MLP(nn.Module):
    def __init__(self,blocks,dropout_setval):
        super(MLP, self).__init__()

        self.init_layer = nn.Linear(6,blocks[0])
        self.init_act = nn.ReLU()
        #[64,128,512,1024,512,128,64,3],
        self.L1 = nn.Linear(blocks[0],blocks[1])  # 64   -> 128
        self.L2 = nn.Linear(blocks[1],blocks[2])  # 128  -> 256
        self.L5 = nn.Linear(blocks[4],blocks[5])  # 256  -> 128
        self.L6 = nn.Linear(blocks[5],blocks[6])  # 128 -> 64
        self.L7 = nn.Linear(blocks[6],6)  # 64 -> 3
        self.act = nn.SELU()
        self.drop = nn.Dropout(p=dropout_setval)

    def forward(self,x):

        # Input process
        x = self.init_layer(x)
        x = self.init_act(x)
        x = self.drop(x)

        x = self.L1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L5(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L6(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L7(x)

        return x
