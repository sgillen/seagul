"""
Very likely this will be turned into a notebook before too long...
"""

import torch

save_dir = "./data/test/"
policy = torch.load(save_dir + "policy")
value_fn = torch.load(save_dir + "value_fn")

policy(torch.rand(4))
