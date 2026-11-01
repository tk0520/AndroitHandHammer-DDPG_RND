import settings

class QLerning():
    def __init__(self, agent):
        self.agent = agent

    def get_q_value(self, observation, next_observation, reward, terminated):
        if terminated:
            y = reward
        else:
            y = reward + settings.DISCOUNT * self.agent.target_evaluate(next_observation)
        
        return y