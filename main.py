import gymnasium_robotics
import gymnasium as gym
import numpy as np
import envpool

from agent.agent import HumanoidAgent
from record.setup import setup_eval_env
from train import start_train
import settings

gym.register_envs(gymnasium_robotics)

train_envs = envpool.make_gymnasium(
    settings.ENV_NAME, 
    num_envs=settings.ENVS_NUM, seed=0, 
    render_mode=None
)
teacher_student_env = envpool.make_gymnasium(
    settings.ENV_NAME, 
    num_envs=settings.ENVS_NUM, seed=0, 
    render_mode=None,
    max_episode_steps=500
)
eval_env = setup_eval_env()

print(train_envs.observation_space)

observation_space = (train_envs.observation_space.shape[0])
action_space = train_envs.action_space.shape[0]

agent = HumanoidAgent(state_space=observation_space, action_space=action_space, action_range=[-1, 1])
start_train(train_envs, teacher_student_env, eval_env, agent)
