# seagul

> This is my rifle. There are many like it, but this one is mine.
> 
>  \- Major General William H. Rupertus 


A generic utility library for deep reinforcement learning. I built this throughout my PhD as sort of a personal swiss-army knife for my research. There are other libraries, big and small, that do similar things, but this one is mine. 

Highlights include:

- Implementations of several deep reinforcement learning algorithms, including [Proximal Policy Optimization](./seagul/rl/ppo),  [Augmented Random Search](./seagul/rl/ars), and [Soft Actor Critic](./seagul/rl/sac). All using a unified interface.

- [Neural network utilities](./seagul/nn.py), Including supervised learning functionality mimicking keras' model.fit in pytorch, and an interface to make various MLPs, RBFs etc that are compatible with the rest of this library.

- [Tests](./seagul/tests/) Known good hyper parameters across different environements for the algorithms. 

- [Custom OpenAI gym environments](./seagul/envs), That I've studied at some point or another in my research. 

- [Experiment utilities](./seagul/rl/run_utils), For saving RL experiments along with meta data, logs, and a mechanism to restore the trained agents.


## Who should use this?

If you just want to get your feet wet with deep RL there are better places to start, I usually recomend [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Most of the value of something like this is that I personally know it inside and out, but that won't be an advantage to someone out. 

If you are extending my previous work or something though you are in the right place.

## Installation
This is a python package, so you can install it with:

```
git clone https://github.com/sgillen/seagul
cd seagul
pip install -e .
```

I think that the tests folder linked above is a good starting point, and many files have a __main__ restricted portion that shows some basic usage. 
