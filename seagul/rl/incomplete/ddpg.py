import numpy as np


# This will probably need to be a circular buffer
class ReplayBuffer:
    def __init__(self, state_size, action_size, buf_size=100000):
        self.state_buf = np.zeros(state_size, buf_size)
        self.action_buf = np.zeros(action_size, buf_size)
        self.circ_index = 0
