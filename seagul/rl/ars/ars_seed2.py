import numpy as np
import torch
import copy
import gym


class MetaArsAgent:
    def __init__(self, meta_seed, init_ars_agent, n_seeds=8, n_top_seeds=2, mean_lookback=10, ars_epochs=10):

        self.init_ars_agent = init_ars_agent
        self.meta_seed = meta_seed
        self.n_seeds = n_seeds
        self.n_top_seeds = n_top_seeds
        self.mean_lookback = mean_lookback
        self.ars_epochs = ars_epochs

    def learn(self, meta_epochs):
        n_children = self.n_seeds//self.n_top_seeds
        np.random.seed(self.meta_seed)
        reward_log = []
        agent_log = []

        top_agents = []
        for _ in range(self.n_top_seeds):
            top_agents.append(copy.deepcopy(self.init_ars_agent))

        for epoch in range(meta_epochs):
            agent_list = []
            rew_list = []
            for agent in top_agents:
                for _ in range(n_children):
                    new_agent = copy.deepcopy(agent)
                    new_agent.seed = np.random.randint(0, 2**32-1, 1)

                    rews = new_agent.learn(self.ars_epochs)

                    agent_list.append(new_agent)

                    r = torch.stack(rews[-self.mean_lookback:])
                    rew_list.append(r.mean())

            top_idx = sorted(range(len(rew_list)), key=lambda k: rew_list[k], reverse=True)[:self.n_top_seeds]
            top_agents = []
            for i in top_idx:
                top_agents.append(agent_list[i])

            agent_log.append(copy.deepcopy(agent_list))

            reward_log.append(max(rew_list))

        return top_agents, reward_log, agent_log


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    import matplotlib.pyplot as plt
    from seagul.nn import MLP
    from seagul.rl.ars.ars_pipe2 import ARSAgent

    env_name = "Walker2d-v2"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size, 0, 0, bias=False)

    import time
    start = time.time()
    init_agent = ARSAgent(env_name, policy, seed=0, n_workers=12, n_delta=32, n_top=16)
    meta_agent = MetaArsAgent(0, init_agent, n_seeds=8, n_top_seeds=1, mean_lookback=3, ars_epochs=25)
    agents, reward_log, agent_log, = meta_agent.learn(20)

    print(time.time() - start)

    plt.plot(reward_log)
    plt.show()

    for agents in agent_log:
        for a in agents:
            plt.plot(a.lr_hist)

    plt.show()

    #env = gym.make(env_name)
    #state_hist, act_hist, returns = do_rollout(env_name, policy)
