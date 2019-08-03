# Sgillen Research

This is a collection of various pieces of software I use for research. 

# Installing Dependencies

I use conda to manage most of my dependencies. To set create your own conda env for use of this code use:

```
conda env create -f environment.yml
```

# Additional Dependencies

Depending on which parts of my code you want to use you may or may not also need to manually install the following

## Mujoco/Mujoco py
http://www.mujoco.org
https://github.com/openai/mujoco-py (also has instructions on how to install mujoco)

Note you need a liscense to use mujoco, but a free one can be obtained without too much pain. either get a no questions asked 30 day trial or a free one year renewable one. If you publish with it you will need to pay up though!

## Open AI baselines
https://github.com/openai/baselines

Note: Make sure you install from the github and not the version that you can obtain by default using pip

## Drake

(http://drake.mit.edu). As well as the code/notes here 
(https://github.com/RussTedrake/underactuated). Both will need to be added to 
your python path, it should look something like

export PYTHONPATH=path_to_drake/lib/python2.7/site-packages:path_to_underactuated/underactuated/src/:${PYTHONPATH}



