from seagul.rl.ars.ars_np_queue import ARSAgent
import numpy as np


class MetaARSAgent:
    def __init__(self, env_name, n_agents=4, n_top_agents=2, epochs_per_meta_update=10, kwargs=None):
        self.env_name = env_name

        assert((n_agents - n_top_agents) % n_top_agents == 0)
        self.n_agents = n_agents
        self.n_top_agents = n_top_agents
        self.epochs_per_meta_update = epochs_per_meta_update

        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        self.seeds = np.random.randint(0,2**32-1,n_agents)
        self.agents = [ARSAgent(env_name, seed, **kwargs) for seed in self.seeds]
        self.ml_hist = []

    def learn(self, meta_steps, verbose=False):
        for meta_steps in range(meta_steps):
            for agent in self.agents:
                agent.learn(self.epochs_per_meta_update, verbose=verbose)

            end_rewards = [np.mean(agent.lr_hist[-5:]) for agent in self.agents]
            sorted_idx = sorted(range(len(end_rewards)), key = lambda k: end_rewards[k], reverse=True)
            sorted_agents = [self.agents[k] for k in sorted_idx]
            top_agents = sorted_agents[:self.n_top_agents]
            bad_agents = sorted_agents[self.n_top_agents:]

            num_agents_to_copy = len(bad_agents)//len(top_agents)

            for i in range(self.n_top_agents):
                print(f"Top agent {sorted_idx[i]} had reward: {np.mean(top_agents[i].lr_hist[-5:])}")

            for i in range(len(top_agents)):
                for j in range(num_agents_to_copy):
                    print(f"copying agent {sorted_idx[i]} into {sorted_idx[len(top_agents) + i*num_agents_to_copy+j]}")
                    bad_agents[i*num_agents_to_copy+j].W = np.copy(top_agents[i].W)

            self.ml_hist.append(top_agents[0].lr_hist[-1])

if __name__ == "__main__":
    meta_agent = MetaARSAgent("Hopper-v2", n_agents=10, kwargs={"step_size": .05, "exp_noise": .05})
    meta_agent.learn(5)