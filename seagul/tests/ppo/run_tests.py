from seagul.tests.ppo.invertedpend_ppo2 import run_and_test as run_inverted_pend
from seagul.tests.ppo.pend_ppo2_var import run_and_test as run_pend
from seagul.tests.ppo.linearz_ppo import run_and_test as run_linearz
from seagul.tests.ppo.linearz2d_ppo import run_and_test as run_linearz_2d

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


def split_results(results):
    rewards = []; finished = []
    for result in results:
        rewards.append(result[0])
        finished.append(result[1])

    return rewards, finished

plt.figure()

plt.show()


if __name__ == "__main__":
    seeds = np.random.randint(0, 2 ** 32, 8)
    pool = Pool(processes=8)

    results = pool.map(run_pend, seeds)
    rewards, finished = split_results(results)
    print(all(finished))

    for reward in rewards:
        plt.plot(reward, alpha=.8)

    results = pool.map(run_pend, seeds)
    rewards, finished = split_results(results)
    print(all(finished))

    for reward in rewards:
        plt.plot(reward, alpha=.8)

    plt.figure()
    for reward in rewards:
        plt.plot(reward, alpha=.8)
