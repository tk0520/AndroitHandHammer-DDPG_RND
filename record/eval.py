import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import imageio
import torch
import time
import sys
import os

from utils import concatenate_observations
import settings

def record_video(record_env, eval_agent):
    obs_dict, info = record_env.reset()
    observation = concatenate_observations(obs_dict)  
    frames, total_reward, done = [], 0, False

    while not done:
        with torch.no_grad():
            action = eval_agent.select_actions(observation, exploration=False)
            action = action.reshape(-1)

        img = record_env.render()          
        frames.append(img)            

        obs_dict, reward, terminated, truncated, info = record_env.step(action)
        observation = concatenate_observations(obs_dict)
        total_reward += reward
        done = terminated or truncated

    print(f"✅ {len(frames)} frames, reward={total_reward}")

def record_stat(episode_avg_rewards, replay_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # --- 1. Reward Plot ---
    ax1.plot(episode_avg_rewards[::50], color='blue', linewidth=1)
    ax1.set_title("Avg Rewards")
    ax1.set_xlabel("Per 50 Episode")
    ax1.set_ylabel("Avg Reward")

    # # Critic Loss plot
    ax2.plot(replay_losses[::50], color='red', linewidth=1)
    ax2.set_title("Loss Per 50 Episode")
    ax2.set_xlabel("Per 50 Episodes")
    ax2.set_ylabel("Loss")
    ax2.set_yscale('log')

    # # Actor Loss plot
    # ax3.plot(actor_losses, color='green', linewidth=1)
    # ax3.set_title("Actor Loss Per Episode")
    # ax3.set_xlabel("Per Episode")
    # ax3.set_ylabel("Loss")
    # ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"./stats/{len(episode_avg_rewards)}_graph.png")
    plt.close()