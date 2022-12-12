import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Define model
class MyNetwork(nn.Module):
    def __init__(self, batch_size, width, deepth=3):
        """模型初始化
        batch_size: samples number of one batch
        width: hidden network's width
        deepth: set the deepth of network, which means how much hidden_layer in the model
        """
        super().__init__()
        self.batch_size = batch_size
        self.input_layer = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(), 
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.out_layer = nn.Linear(width, 1)
        self.deepth = deepth
    
    def forward(self, x):

        x = self.input_layer(x)
        for _ in range(self.deepth):
            x = self.hidden_layer(x)
        logits = self.out_layer(x)
        return logits
        
        