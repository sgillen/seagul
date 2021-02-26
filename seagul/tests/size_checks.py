import numpy as np
import torch
from seagul.nn import MLP
from seagul.rl.ppo import PPOModel
from seagul.rl.sac import SACModel


dtype=np.float32


def check_ppo_model(model, obs, act_size, val_size):

    act, val, _, logp = model.step(obs)
    assert (act.shape == act_size)
    assert (val.shape == val_size)
    assert (logp.shape == act_size)

    act, logp = model.select_action(obs)
    assert (act.shape == act_size)

    logp = model.get_logp(obs, act)
    assert (logp.shape == act_size)
    assert (logp.shape == act_size)


def check_sac_model(model, obs, act_size, val_size):

    act, val, _, logp = model.step(obs)
    assert (act.shape == act_size)
    assert (val.shape == val_size)
    assert (logp.shape == act_size)

    act, logp = model.select_action(obs, torch.ones(act_size))
    assert (act.shape == act_size)


# Single output MLP size check
# =================================================================
net = MLP(4,1,2,12)
obs = np.zeros(4, dtype=dtype)
assert net(obs).shape == torch.Size([1])


obs = np.zeros((1,4), dtype=dtype)
assert net(obs).shape == torch.Size([1,1])

obs = np.zeros((100,4), dtype=dtype)
assert net(obs).shape == torch.Size([100,1])

print("Single output MLP good")

# Multiple output MLP size check
# =================================================================
net = MLP(4,4,2,12)
obs = np.zeros(4, dtype=dtype)
assert net(obs).shape == torch.Size([4])

obs = np.zeros((1,4), dtype=dtype)
assert net(obs).shape == torch.Size([1, 4])

obs = np.zeros((100,4), dtype=dtype)
assert net(obs).shape == torch.Size([100, 4])

print("Multiple output MLP good")

# Single output PPO size check, fixed variance
# =================================================================
model = PPOModel(
    policy = MLP(4, 1, 2, 12),
    value_fn = MLP(4, 1, 2, 12),
)

obs = np.zeros(4, dtype=dtype)
check_ppo_model(model, obs, torch.Size([1]), torch.Size([1]))

obs = np.zeros((1,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([1,1]), torch.Size([1,1]))

obs = np.zeros((100,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([100,1]), torch.Size([100,1]))

print("Single output PPO model good")


# Multiple output PPO size check
# =================================================================


model = PPOModel(
    policy = MLP(4, 4, 2, 12),
    value_fn = MLP(4, 1, 2, 12),
    fixed_std = True
)

obs = np.zeros(4, dtype=dtype)
check_ppo_model(model, obs, torch.Size([4]), torch.Size([1]))

obs = np.zeros((1,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([1,4]), torch.Size([1,1]))

obs = np.zeros((100,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([100,4]), torch.Size([100,1]))

model = PPOModel(
    policy = MLP(4, 8, 2, 12),
    value_fn = MLP(4, 1, 2, 12),
    fixed_std = False
)

obs = np.zeros(4, dtype=dtype)
check_ppo_model(model, obs, torch.Size([4]), torch.Size([1]))

obs = np.zeros((1,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([1,4]), torch.Size([1,1]))

obs = np.zeros((100,4), dtype=dtype)
check_ppo_model(model, obs, torch.Size([100,4]), torch.Size([100,1]))


print("Multiple output PPO model good")



# Single output SAC size check
# =================================================================
model = SACModel(
    policy = MLP(4, 2, 2, 12),
    value_fn = MLP(4, 1, 2, 12),
    q1_fn=MLP(5, 1, 2, 12),
    q2_fn=MLP(5, 1, 2, 12),
    act_limit=1
)


obs = np.zeros(4, dtype=dtype)
check_sac_model(model, obs, torch.Size([1]), torch.Size([1]))

obs = np.zeros((1,4), dtype=dtype)
check_sac_model(model, obs, torch.Size([1,1]), torch.Size([1,1]))

obs = np.zeros((100,4), dtype=dtype)
check_sac_model(model, obs, torch.Size([100,1]), torch.Size([100,1]))

print("Single output SAC model good")


# Multiple output SAC size check
# =================================================================

model = SACModel(
    policy = MLP(4, 8, 2, 12),
    value_fn = MLP(4, 1, 2, 12),
    q1_fn=MLP(8, 1, 2, 12),
    q2_fn=MLP(8, 1, 2, 12),
    act_limit=1
)



obs = np.zeros(4, dtype=dtype)
check_sac_model(model, obs, torch.Size([4]), torch.Size([1]))

obs = np.zeros((1,4), dtype=dtype)
check_sac_model(model, obs, torch.Size([1,4]), torch.Size([1,1]))

obs = np.zeros((100,4), dtype=dtype)
check_sac_model(model, obs, torch.Size([100,4]), torch.Size([100,1]))

print("Multiple output SAC model good")
