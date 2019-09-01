"""
Utility functions for running and playing back reinforcement learning algos.
This also supports running most algorithms from openAI baselines

Assumes you are using the modified version hosted here: https://github.com/sgillen/baselines which properly saves the
VecNormalize variables for mujoco environments.

The advantage of using this over the command line interface is that these functions automatically keep track of meta
data (what arguments were used for a run_util, how long did the run_util take, what were you trying to accomplish), and takes
care of loading a trained model just by specifying the name you saved it with

"""

import baselines.run

from seagul.nn import MLP
from seagul.rl.ppo import ppo

import torch
import torch.nn as nn

import dill
import pickle
import time, datetime, json, yaml
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_and_save_bs(arg_dict, run_name=None, description=None, base_path="/data/", backend="baselines"):

    """
    Launches an openAI baselines algorithm and records the results

    If you don't pass run_name or description this function will call input, blocking execution

    This function is my entry point to the openAI baselines command line. I like to be able to programmatically specify
    the arguments to use, and I'd like to be able to keep track of other meta data, plus log everything to the same
    place. Furthermore I'd like to log my data in a format where I can just point to a save file and reconstruct the
    network structure needed to play the agent out in the environment or test it in a new one

    Args:
        arg_dict: dictionary of arguments, you need to use the exact name that openAI uses
        run_name: name to save the run_util under
        description: brief description of what you were trying to do with the run_util

    Returns:
        Does not return anything

    Example:

        from run_rl.run_baselines import run_baselines

        arg_dict = {
        'env' : 'su_cartpole-v0',
        'alg' : 'ppo2',
        'network' : 'mlp',
        'num_timesteps' : '2e4',
        'num_env' : '1'
        }

        run_baselines(arg_dict, run_name='test2', description='')

    """

    if run_name is None:
        run_name = input("please enter a name for this run: ")

    if description is None:
        description = input("please enter a brief description of the run: ")

    save_base_path = os.getcwd() + base_path
    save_dir = save_base_path + run_name + "/"
    save_path = save_dir + "saved_model"
    arg_dict["save_path"] = save_path

    baselines_path = baselines.run.__file__
    os.environ["OPENAI_LOGDIR"] = save_dir
    os.environ["OPENAI_LOG_FORMAT"] = "stdout,csv,tensorboard"

    argv_list = [baselines_path]  # first argument is the path of baselines.run_util

    for arg, value in arg_dict.items():
        argv_list.append("--" + str(arg) + "=" + value)

    start_time = time.time()
    baselines.run.main(argv_list)
    runtime = time.time() - start_time

    datetime_str = str(datetime.datetime.today())
    datetime_str = datetime_str.replace(" ", "_")
    runtime_str = str(datetime.timedelta(seconds=runtime))

    with open(save_dir + "info.json", "w") as outfile:
        json.dump(
            {
                "args": arg_dict,
                "metadata": {"date_time": datetime_str, "total runtime": runtime_str, "description": description},
            },
            outfile,
            indent=4,
        )


def run_clean_ppo(run_name = None, description = None, base_path = '/data/'):
    """
    Launches seaguls ppo2 and save the results without clutter

    If you don't pass run_name or description this function will call input, blocking execution
    """


    if run_name is None:
        run_name = input("please enter a name for this run_util: ")

    if description is None:
        description = input("please enter a brief description of the run_util: ")

    save_base_path = os.getcwd() + base_path
    save_dir = save_base_path + run_name + "/"
    save_path = save_dir + "saved_model"

    start_time = time.time()

    ## init policy, valuefn
    input_size = 4
    output_size = 1
    layer_size = 64
    num_layers=3
    activation=nn.ReLU

    torch.set_default_dtype(torch.double)

    arg_dict = {
        'env_name' : 'su_cartpole-v0',
        'num_epochs' : 5,
        'action_var' : .1,
    }

    net_dict = {
         'policy': MLP(input_size, output_size, num_layers, layer_size, activation),
         'value_fn': MLP(input_size, 1, num_layers, layer_size, activation),
    }

    t_policy, t_val, rewards, var_dict = ppo(**arg_dict,  **net_dict)
    runtime = time.time() - start_time

    datetime_str = str(datetime.datetime.today())
    datetime_str = datetime_str.replace(" ", "_")
    runtime_str = str(datetime.timedelta(seconds=runtime))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "info.yaml", "w") as outfile:
        yaml.dump(
            {
                "args": arg_dict,
                "metadata": {"date_time": datetime_str, "total runtime": runtime_str, "description": description},
            },
            outfile,
            indent=4,
        )


    with open(save_dir + "workspace", "wb") as outfile:
        for thing in var_dict:
            try:
                pickle.dump(thing, outfile)
            except Exception as e:
                print(e, thing)




def load_model(save_path, backend="baselines"):

    """
    Loads and plays back a trained model.


    You must either specify a relative directory with the ./notation, or the absolute path. 
    However absolute paths only work with mac or Linux.

    Parameters:
        save_path: a string with the name you want to load. You probably are running this file that looks like: ~/work/data/run1/run1
        to load it provide the string './data/run1'

        backend: a string, either 'baselines' or 'seagul' depending on which implementation you used

    Returns:
       returns the model and the environment


    Example:
        from run_rl.run_baselines import play_baselines
        model, env = play_baseline('./data/run1')
    """

    if save_path[-1] == "/":
        save_path = save_path[:-1]

    if save_path.split("/")[1] == "home" or save_path.split("/")[1] == "User":
        save_base_path = save_path
    else:
        save_base_path = os.getcwd() + save_path.split(".")[1]

    run_name = save_path.split("/")[-1]
    # load_dir = save_base_path + run_name + 'info.json'
    # arg_dict['load_path']

    if backend == "baselines":
        with open(save_base_path + "/" + "info.json", "r") as outfile:
            data = json.load(outfile)

        arg_dict = data["args"]
        arg_dict["num_timesteps"] = "0"
        arg_dict["num_env"] = "1"
        del arg_dict["save_path"]
        arg_dict["load_path"] = save_base_path + "/" + "saved_model"

        baselines_path = baselines.run.__file__
        argv_list = [baselines_path]  # first argument is the path of baselines.run_util

        for arg, value in arg_dict.items():
            argv_list.append("--" + str(arg) + "=" + value)

        argv_list.append("--play")

        model, env = baselines.run.main(argv_list)

        return model, env

    elif backend == "seagul":
        with open(save_base_path + "/" + "info.json", "r") as outfile:
            data = yaml.load(outfile)

        # TODO need to make model from disparate networks... shit...

        #with open(save_base_path + "/" + "info.json", "r") as outfile:
        #    data =
    else:
        raise ValueError("unrecognized backend: ", backend)


if __name__ == "__main__":
    run_clean_ppo('test2', 'test test')