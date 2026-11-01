import numpy as np

def concatenate_observations(obs):
    return obs
    try:
        return np.concatenate([
            obs['observation'],
            obs['achieved_goal'],
            obs['desired_goal']
        ], axis=1)
    except:
        return np.concatenate([
            obs['observation'],
            obs['achieved_goal'],
            obs['desired_goal']
        ])

def get_observation_shape(obs):
    try:
        return obs['observation'].shape[0] + obs['achieved_goal'].shape[0] + obs['desired_goal'].shape[0]
    except:
        return obs.shape[0]