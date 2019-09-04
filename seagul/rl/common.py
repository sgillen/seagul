import torch

# can make this faster I think?
def discount_cumsum(rewards, discount):
    future_cumulative_reward = 0
    cumulative_rewards = torch.empty_like(torch.as_tensor(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + discount * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards
