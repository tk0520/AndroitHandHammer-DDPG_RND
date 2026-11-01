from torchrl.modules import BatchRenorm1d, NoisyLinear
from torch.nn.utils import weight_norm
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
 
import settings
 
class DDPGActorModel(nn.Module):
    def __init__(self, state_space, action_space, device):
        super().__init__()  
 
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
 
        self.policy = nn.Sequential(
            NoisyLinear(state_space, 256, device=self.device), 
            nn.LayerNorm(256), 
            nn.LeakyReLU(),
            NoisyLinear(256, 512, device=self.device),
            nn.LayerNorm(512), 
            nn.LeakyReLU(),
            NoisyLinear(512, 1024, device=self.device),
            nn.LayerNorm(1024), 
            nn.LeakyReLU(),
            NoisyLinear(1024, 2048, device=self.device),
            nn.LayerNorm(2048), 
            nn.LeakyReLU(),
            NoisyLinear(2048, 4056, device=self.device),
            nn.LayerNorm(4056), 
            nn.LeakyReLU(),
            NoisyLinear(4056, 2048, device=self.device),
            nn.LayerNorm(2048), 
            nn.LeakyReLU(),
            NoisyLinear(2048, 1024, device=self.device),
            nn.LayerNorm(1024), 
            nn.LeakyReLU(),
            NoisyLinear(1024, 512, device=self.device),
            nn.LayerNorm(512), 
            nn.LeakyReLU(),
            NoisyLinear(512, 256, device=self.device),
            nn.LayerNorm(256), 
            nn.LeakyReLU(),
            NoisyLinear(256, action_space, device=self.device),
            nn.Tanh()
        )
 
    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.to(self.device)
 
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
 
    def forward(self, x):
        x = self.preprocess(x)
        return self.policy(x)

class DDPGCriticModel(nn.Module): 
    def __init__(self, state_space, action_space, device):
        super().__init__()  
        self.device = device
        self.state_space = state_space
        self.action_space = action_space
 
        self.value = nn.Sequential(
            nn.Linear(state_space + action_space, 256),  
            nn.LayerNorm(256), 
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512), 
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), 
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.to(self.device)
 
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def forward(self, x):
        x = self.preprocess(x)
        return self.value(x)
    

 