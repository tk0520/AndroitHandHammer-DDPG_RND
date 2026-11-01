from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium_robotics
import gymnasium as gym
import numpy as np
import sys

from utils import concatenate_observations
import settings

def setup_eval_env():
    gym.register_envs(gymnasium_robotics)
    eval_env = gym.make(
        f'{settings.ENV_NAME}', render_mode="rgb_array", 
        camera_id=0, max_episode_steps=500
    )
    eval_env = setup_video_record(eval_env)
    return eval_env
    

def setup_video_record(eval_env):
    eval_env = RecordVideo(
        eval_env,
        video_folder="./videos",
        episode_trigger=lambda x: True,
        name_prefix=settings.ENV_NAME,
        fps=35
    )
    return eval_env

def record_video(eval_env, eval_agent):
    observation, info = eval_env.reset()
    observation = concatenate_observations(observation)
    done = False

    print("=" * 100)
    print("Info:", info)
    print("Evaluation Env Start...!")

    while not done:
        policy = eval_agent.get_policy(observation)
        action = eval_agent.get_actions(policy)

        observation, reward, terminated, truncated, info = eval_env.step(action)
        observation = concatenate_observations(observation)
        done = terminated or truncated

    print("Evaluation Env End...!")
    print("=" * 100)