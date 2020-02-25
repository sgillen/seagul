import numpy as np
import array
import time

dtype="f" # need to change d->f this for float32
ep_steps = 200
obs_size  = 4
act_size = 1


def calc_rewards(obs, act):
    return np.ones(ep_length,1)

def do_simulink_rollout(env, model):
    # Env is completley ignored but we need to match the signature of the normal do_rollout

    
    
    write_net("policy", model.policy)
    write_net("value_fn" , model.value_fn)

    with open("update_ready", "w") as f:
        f.write("1")

    # this is terrible I know but the only way to get anything out of simulink is via disk backed file IO

    ep_done = "0"
    while(ep_done == "0"):
        with open("episode_done", "r") as f:
            ep_done = f.read()
            time.sleep(.1)


    print("here")
    
    obs = read_array('obs' , ep_steps, obs_size)
    act = read_array('act' , ep_steps, act_size)
    done = np.zeros(ep_steps,1)
    rews = calc_rewards(obs, act)

    return obs, act, done, rews
    
# This relies on the net_dict being ordered
def write_net(prefix, net_dict_vals, num_layers = 4):
    weight_names = ["wght" + str(num) + ".dat" for num in range(num_layers)]
    bias_names   = ["bias" + str(num) + ".dat" for num in range(num_layers)]
    file_names = [val for pair in zip(weight_names, bias_names) for val in pair]
    
    for name, value in zip(file_names, net_dict_vals):
        write_layer(prefix + name, value)

    return

    
def read_array(file_name, dim1, dim2):
    f = open(file_name, "rb")
    a = array.array(dtype) 
    A = np.zeros((dim1, dim2))

    size = dim1 * dim2
    a.fromfile(f, size)
    for i in range(dim2):
        A[:, i] = a[i * dim1 : i * dim1 + dim1]

    return A


def write_layer(file_name, A):
    f = open(file_name, "wb")
    a = array.array(dtype)

    for entry in A.flatten():
        a.append(entry)

    a.tofile(f)
    return


# Don't actually need this eh? maybe one day
def read_net(prefix, net, num_layers = 4):
    weight_names = ["wght.dat" + str(num) for num in range(num_layers)]
    bias_names   = ["bias.dat" + str(num) for num in range(num_layers)]
    file_names = [val for pair in zip(weight_names, bias_names) for val in pair]

    net_dict = net.state_dict()
    
    for file_name, layer_name, value in zip(file_names, net_dict.keys(), net.dict.values()):
        net_dict[layer_name] = read_weights(prefix + file_name, value)

    return

    
