from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
import copy
 
from agent.model import DDPGActorModel, DDPGCriticModel
from agent.exploration import RND
from agent.computation import QLerning
from agent.memory import ReplayMemory
import settings
 
class HumanoidAgent:
    def __init__(self, state_space, action_space, action_range):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space
        self.action_range = action_range

        self.actor_model = DDPGActorModel(
            self.state_space, 
            self.action_space, 
            self.device
        ).to(self.device)
        self.old_actor_model = copy.deepcopy(self.actor_model).to(self.device)

        self.critic_model_1 = DDPGCriticModel(
            self.state_space, 
            self.action_space, 
            self.device
        ).to(self.device)
        self.old_critic_model_1 = copy.deepcopy(self.critic_model_1)

        self.critic_model_2 = DDPGCriticModel(
            self.state_space,
            self.action_space,
            self.device
        ).to(self.device)
        self.old_critic_model_2 = copy.deepcopy(self.critic_model_2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), 
            lr=settings.LEARNING_RATE
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_model_1.parameters(), 
            lr=0.001
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_model_2.parameters(), 
            lr=0.001
        )

        self.rnd = RND(state_space=self.state_space, device=self.device) 
        self.replay_memory = ReplayMemory()
        self.computation = QLerning(self)     

        self.total_updates = 0
        self.policy_delay = 2
        self.reset_counter = 0
        
    def select_actions(self, observations, exploration=True):
        low  = torch.as_tensor(self.action_range[0], dtype=torch.float32, device=self.device)
        high = torch.as_tensor(self.action_range[1], dtype=torch.float32, device=self.device)
        actions = self.actor_model(observations)

        if exploration:
            actions = torch.normal(mean=actions, std=torch.full_like(actions, settings.EXPLORATION_STD))
        
        actions = actions.clamp(low, high)
        return actions.detach().cpu().numpy().astype(np.float32)
        
    def select_old_actions(self, observations):
        range_low  = torch.as_tensor(
            self.action_range[0], dtype=torch.float32, device=self.device
        )
        range_high = torch.as_tensor(
            self.action_range[1], dtype=torch.float32, device=self.device
        )

        actions = self.old_actor_model(observations)
        actions = actions.clamp(min=range_low, max=range_high)
        return actions 
 
    def current_evaluate_1(self, observations):
        return self.critic_model_1(observations)

    def current_evaluate_2(self, observations):
        return self.critic_model_2(observations)
 
    def old_evaluate_1(self, observations):
        return self.old_critic_model_1(observations)

    def old_evaluate_2(self, observations):
        return self.old_critic_model_2(observations)
    
    def soft_update(self, target, source, tau=0.005):
        for target, source in zip(target.parameters(), source.parameters()):
            target.data.copy_(tau * source.data + (1.0 - tau) * target.data)

    def reset_noise(self):
        for module in self.actor_model.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

    def reset_weight(self):
        def weights_init(model):
            if isinstance(model, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(model.weight.data)
    
                if model.bias is not None:
                    torch.nn.init.zeros_(model.bias.data)

        if self.reset_counter % settings.RESET_INTERVAL == 0:
            self.actor_model.apply(weights_init)
            self.critic_model_1.apply(weights_init)
            self.critic_model_2.apply(weights_init)
            self.old_actor_model.apply(weights_init)
            self.old_critic_model_1.apply(weights_init)
            self.old_critic_model_2.apply(weights_init)
            self.reset_counter += 1
            print("RESET YEAH")
        else:
            self.reset_counter += 1

    def update(self, samples, mode=None):
        observations, actions, rewards, next_observations, terminated = zip(*samples)

        observations = torch.tensor(
            np.array([obs[0] if isinstance(obs, tuple) else obs for obs in observations]),
            dtype=torch.float32
        ).to(self.device)
        next_observations = torch.tensor(
            np.array([obs[0] if isinstance(obs, tuple) else obs for obs in next_observations]),
            dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            np.array([action[0] if isinstance(action, tuple) else action for action in actions]),
            dtype=torch.float32
        ).to(self.device)
        rewards = torch.tensor(
            np.array([reward[0] if isinstance(reward, tuple) else reward for reward in rewards]),
            dtype=torch.float32
        ).to(self.device)
        terminated = torch.tensor(
            np.array([term[0] if isinstance(term, tuple) else term for term in terminated]),
            dtype=torch.float32
        ).to(self.device)

        # Intrinsic Rewards ===================================================
        intrinsic_reward, pseudo_count = self.rnd.get_intrinsic_reward(
            next_observations, beta=settings.RND_WEIGHT
        )
        rewards = rewards + intrinsic_reward.squeeze(-1)
        # Intrinsic Rewards ===================================================

        # Critic Loss ===============================================================
        with torch.no_grad():
            rewards = rewards.unsqueeze(1)
            terminated = terminated.unsqueeze(1)
            next_actions = self.select_old_actions(next_observations)

            target_q1 = self.old_evaluate_1(torch.cat((next_observations, next_actions), dim=1))
            target_q2 = self.old_evaluate_2(torch.cat((next_observations, next_actions), dim=1))

            next_q_values = torch.min(target_q1, target_q2)
            targets_critic = rewards + (1 - terminated) * settings.DISCOUNT * next_q_values

        current_q_1 = self.current_evaluate_1(torch.cat((observations, actions), dim=1))
        current_q_2 = self.current_evaluate_2(torch.cat((observations, actions), dim=1))

        critic_1_loss = torch.mean((targets_critic - current_q_1) ** 2)
        critic_2_loss = torch.mean((targets_critic - current_q_2) ** 2)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()

        clip_grad_norm_(self.critic_model_1.parameters(), max_norm=1.0)
        clip_grad_norm_(self.critic_model_2.parameters(), max_norm=1.0)

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        # Critic Loss ===============================================================

        # Delayed Actor Loss + Soft Update ==========================================
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:

            # Actor Loss ------------------------------------------------------------
            new_actions = self.actor_model(observations)
            new_q_values = self.current_evaluate_1(torch.cat((observations, new_actions), dim=1))
            avg_grad_objective = -new_q_values.mean()

            self.actor_optimizer.zero_grad()
            avg_grad_objective.backward()
            clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            # Actor Loss ------------------------------------------------------------

            # Soft Update -----------------------------------------------------------
            self.soft_update(self.old_actor_model, self.actor_model)
            self.soft_update(self.old_critic_model_1, self.critic_model_1)
            self.soft_update(self.old_critic_model_2, self.critic_model_2)
            # Soft Update -----------------------------------------------------------

            self.total_updates = 0
        # Delayed Actor Loss + Soft Update ==========================================

    def update_uncertainty(self, samples):
        self.rnd.train(samples)
