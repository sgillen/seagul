import numpy as np
import torch

X = []
Y = []
dtype = torch.long

goal_state = np.array([np.pi / 2, 0, 0, 0])
ep_obs = torch.randn((10, 4))
ep_obs[-1] = torch.as_tensor(goal_state)
ep_path = torch.zeros(10, dtype=dtype)
ep_path[5:] = torch.ones(1)

ep_err = ((ep_obs[-1, :] - goal_state) ** 2).sum().sqrt()
if ep_path.sum() != 0:
    reverse_obs = np.flip(ep_obs.numpy(), 0).copy()
    reverse_obs = torch.from_numpy(reverse_obs)

    reverse_path = np.flip(ep_path.numpy(), 0).copy()
    reverse_path = torch.from_numpy(reverse_path)
    for path, obs in zip(reverse_path, reverse_obs):
        if not path:
            break
        else:
            X.append(obs)
            if ep_err < 2:
                Y.append(torch.ones(1, dtype=dtype))
            else:
                Y.append(torch.zeros(1,dtype=dtype))




goal_state = np.array([np.pi / 2, 0, 0, 0])
ep_obs = torch.randn((10, 4))
ep_obs[-1] = torch.as_tensor(goal_state+10)
ep_path = torch.zeros(10, dtype=dtype)
ep_path[5:] = torch.ones(1)

ep_err = ((ep_obs[-1, :] - goal_state) ** 2).sum().sqrt()

if ep_path.sum() != 0:
    reverse_obs = np.flip(ep_obs.numpy(), 0).copy()
    reverse_obs = torch.from_numpy(reverse_obs)

    reverse_path = np.flip(ep_path.numpy(), 0).copy()
    reverse_path = torch.from_numpy(reverse_path)

    for path, obs in zip(reverse_path, reverse_obs):
        if not path:
            break
        else:
            X.append(obs)
            if ep_err < 2:
                Y.append(torch.ones(1, dtype=dtype))
            else:
                Y.append(torch.zeros(1,dtype=dtype))


