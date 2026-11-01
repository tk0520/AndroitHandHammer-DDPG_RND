import numpy as np

from record.eval import record_video, record_stat
from utils import concatenate_observations
import settings


def start_train(train_envs, teacher_student_env, eval_env, agent):
    episodes = 0
    steps = 0

    episode_avg_rewards = []
    episode_avg_losses = []
    episode_rewards = []
    replay_losses = []

    current_rewards = np.zeros(settings.ENVS_NUM)
    observations, _ = train_envs.reset()
    observations = concatenate_observations(observations)

    while episodes < settings.EPISODES:
        actions = agent.select_actions(observations)
        next_observations, rewards, terminated, truncated, _ = train_envs.step(actions)  # D: real train env
        next_observations = concatenate_observations(next_observations)

        for i in range(settings.ENVS_NUM):
            done = terminated[i] or truncated[i]
            current_rewards[i] += rewards[i]
            steps += 1

            experience = (
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                terminated[i]
            )
            agent.replay_memory.store(experience)

            if done:
                print(f"Env[{i}] completed:", current_rewards[i])
                print("Completed Episodes:", episodes)
                print("Memory Size:", len(agent.replay_memory))

                episode_rewards.append(current_rewards[i])
                last_10_avg_rewards = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                episode_avg_rewards.append(last_10_avg_rewards)

                current_rewards[i] = 0
                episodes += 1

                if episodes % settings.VIDEO_RECORD_INTERVAL == 0:
                    record_video(eval_env, agent)
         
        if len(agent.replay_memory) > settings.INITIAL_MEMORY:
            n_updates = settings.ENVS_NUM // settings.TRAIN_FREQ
            for _ in range(n_updates):
                samples = agent.replay_memory.get_samples()
                loss = agent.update(samples)

                agent.update_uncertainty(samples)
                replay_losses.append(loss)

        observations = next_observations
