# seagul
A utility library. I collected most of the software I use across projects in my research and put it here. 

Highlights include:

- Implementations of several deep reinforcement learning algorithms, including [Proximal Policy Optimization](./seagul/rl/ppo),  [Augmented Random Search](./seagul/rl/ars), and [Soft Actor Critic](./seagul/rl/sac). Along with some novel modifications made as part of my research.

- [Neural network utilities](./seagul/nn.py), including supervised learning functionality mimicking keras' model.fit in pytorch, and an interface to make various MLPs, RBFs etc that are compatible with the rest of this library.

- [Custom OpenAI gym environments](./seagul/envs), that I've studied at some point or another in my research. 

- [Experiment utilities](./seagul/rl/run_utils), for saving RL experiments along with meta data, logs, and a mechanism to restore the trained agents.

And lots of other little odds and ends ([check out all my abandoned projects!](https://github.com/sgillen/misc)). 

## Who should use this?

Probably no one unless your working directly with me, this is meant as a personal swiss-army knife for my research. Still I think it's useful to check out a few implementations of RL algorithms when you are implementing your own. 

## Installation
This is a python package, so you could install it (in some sort of virtual environment!!) with:

```
git clone https://github.com/sgillen/seagul
cd seagul
pip install -e .
```
 
You can install the requirements for the core functionality with:

```
pip install -r requirements.txt
```

