from seagul.nn import fit_model, MLP
from seagul.rl.models import SACModel
import seagul.envs
import torch
import torch.nn as nn
import time
import gym

input_size = 4
output_size = 1
layer_size = 256
num_layers = 2
activation = nn.ReLU

device = 'cpu'

policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation).to(device)
value_fn = MLP(input_size, 1, num_layers, layer_size, activation).to(device)
q1_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation).to(device)
q2_fn = MLP(input_size + output_size, 1, num_layers, layer_size, activation).to(device)
model = SACModel(policy, value_fn, q1_fn, q2_fn, 25)

env = gym.make('su_acrobot-v0')

def do_rollout(env, model, num_steps):
    acts_list = []
    obs1_list = []
    obs2_list = []
    rews_list = []
    done_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs = env.reset()
    done = False
    cur_step = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs1_list.append(obs.clone())

        noise = torch.randn(1, act_size)
        act, _ = model.select_action(obs.reshape(1, -1).to(device), noise.to(device))
        act = act.detach()

        obs, rew, done, _ = env.step(act.cpu().numpy().reshape(-1))
        obs = torch.as_tensor(obs, dtype=dtype).detach()

        acts_list.append(torch.as_tensor(act.clone(), dtype=dtype))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        obs2_list.append(obs.clone())

        if cur_step < num_steps:
            done_list.append(torch.as_tensor(done))
        else:
            done_list.append(torch.as_tensor(False))

        cur_step += 1

    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_obs2 = torch.stack(obs2_list)
    ep_done = torch.stack(done_list).reshape(-1, 1)

    return (ep_obs1, ep_obs2, ep_acts, ep_rews, ep_done)


start = time.time()
for _ in range(100):
    do_rollout(env, model, 500)
print(time.time() - start)


#
# num_samples = 200000
# num_inputs = 4
# num_outputs = 1
# X = torch.randn(num_samples, num_inputs)
# Y = torch.randn(num_samples, num_outputs)
# torch.set_default_dtype(torch.float32)
#
# net1 = MLP(num_inputs, num_outputs, layer_size=256, num_layers=2).to('cuda:0')
# start = time.time()
# fit_model(net1, X, Y, 10, batch_size=4096, use_cuda=True)
# run_time = time.time() - start
# print(run_time)
# print(run_time/num_samples)
#
#
# net1 = MLP(num_inputs, num_outputs, layer_size=256, num_layers=2).to('cuda:0')
# start = time.time()
# fit_model(net1, X, Y, 10, batch_size=8192, use_cuda=True)
# run_time = time.time() - start
# print(run_time)
# print(run_time/num_samples)
#
#
#
# net1 = MLP(num_inputs, num_outputs, layer_size=256, num_layers=2)
# start = time.time()
# fit_model(net1, X, Y, 10, use_cuda=False)
# print(time.time() - start)
#
#
