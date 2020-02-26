import numpy as np
import torch
from torch.distributions import Normal
import time

means = torch.randn(4)
sigma = torch.exp(torch.randn(4))

start = time.time()
for _ in range(1000):
    m = Normal(loc=means, scale=torch.ones_like(means) * sigma)
    action = m.sample()
    logprob1 = m.log_prob(action)
    action = action.detach()
print("method1:", time.time() - start)
print(logprob1)

start = time.time()
for _ in range(1000):
    action = sigma*torch.randn(means.shape[0]) + means
    logprob2 = -((action - means) ** 2) / (2 * sigma**2) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))

print("method2:", time.time() - start)
print(logprob2)