from torch.optim import Adam
from torch import nn
import numpy as np
import torch

import settings

class RND:
    def __init__(self, state_space, device, feature_dim=128):
        self.target = nn.Sequential(
            nn.Linear(state_space, 256), nn.SELU(),
            nn.Linear(256, feature_dim)
        ).to(device)
        self.target.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(state_space, 256), nn.SELU(),
            nn.Linear(256, 256), nn.SELU(),
            nn.Linear(256, feature_dim)
        ).to(device)

        self.optimizer = Adam(self.predictor.parameters(), lr=settings.LEARNING_RATE)
        self.device = device

        self.observation_mean = None
        self.observation_std = None

    def normalize(self, x):
        if self.observation_mean is None: return x
        return torch.clamp((x - self.observation_mean) / (self.observation_std + 1e-8), -5, 5)

    def update(self, observation):  # running mean/std 갱신
        mean = observation.mean(0)
        std = observation.std(0)

        if self.observation_mean is None:
            self.observation_mean, self.observation_std = mean, std
        else:
            self.observation_mean = 0.99 * self.observation_mean + 0.01 * mean
            self.observation_std  = 0.99 * self.observation_std  + 0.01 * std

    def get_intrinsic_reward(self, x, beta=0.1):
        with torch.no_grad():
            x = self.normalize(x)
            error = ((self.predictor(x) - self.target(x)) ** 2).mean(dim=1, keepdim=True)

        pseudo_count = 1.0 / (error + 1e-8)
        intrinsic_reward = beta / torch.sqrt(pseudo_count + 1e-8)  # = beta * sqrt(error)
        return intrinsic_reward, pseudo_count

    def train(self, samples):
        observations = torch.tensor(
            np.array([sample[0] for sample in samples]), dtype=torch.float32
        ).to(self.device)
        
        self.update(observations)
        x = self.normalize(observations)
        
        error = ((self.predictor(x) - self.target(x).detach()) ** 2).mean()
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()