import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from tqdm import trange

"""
Utility functions for seagul, all vaguely related to neural networks
"""

def fit_model(
    model,
    state_train,
    action_train,
    num_epochs,
    learning_rate=1e-2,
    batch_size=32,
    shuffle=True,
    loss_fn=torch.nn.MSELoss(),
    use_tqdm=True,
):
    """
    Trains a pytorch module model to predict actions from states for num_epochs passes through the dataset.

    This is used to do a (relatively naive) version of behavior cloning
    pretty naive (but fully functional) training loop right now, will want to keep adding to this and will want to
    eventually make it more customizable.

    The hope is that this will eventually serve as a keras model.fit function, but customized to our needs.


    Attributes:
        model: pytorch module implementing your controller
        states_train numpy array (or pytorch tensor) of states (inputs to your network) you want to train over
        action_train: numpy array (or pytorch tensor) of actions (outputs of the network)
        num_epochs: how many passes through the dataset to make
        learning_rate: initial learning rate for the adam optimizer

    Returns:
        Returns a list of average losses per epoch
        but note that the model is trained in place!!


    Example:
        model = nn.Sequential(
            nn.Linear(4,12),
            nn.ReLU(),
            nn.Linear(12,12),
            nn.ReLU(),
            nn.Linear(12,1)
            )

        states = np.random.randn(100,4)
        actions = np.random.randn(100,1)

        loss_hist = fit_model(model,states, actions, 200)
    """
    # Check if GPU is available , else fall back to CPU
    # TODO this might belong in module body
    use_cuda = torch.cuda.is_available()
    if use_tqdm:
        range_fn = trange
    else:
        range_fn = range

    # device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")
    state_tensor = torch.as_tensor(state_train)  # make sure that our input is a tensor
    action_tensor = torch.as_tensor(action_train)
    training_data = data.TensorDataset(state_tensor, action_tensor)
    training_generator = data.DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)

    # action_size = action_train.size()[1]

    loss_hist = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range_fn(num_epochs):
        epoch_loss = 0

        for local_states, local_actions in training_generator:

            # Transfer to GPU (if GPU is enabled, else this does nothing)
            local_states, local_actions = (local_states.to(device), local_actions.to(device))

            # predict and calculate loss for the batch
            action_preds = model(local_states)
            loss = loss_fn(action_preds, local_actions)
            epoch_loss += loss.detach()  # only used for metrics

            # do the normal pytorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # after each epoch append the average loss
        loss_hist.append(epoch_loss.numpy() / len(state_train))

    return loss_hist


def policy_render_loop(policy, env, select_action):

    """
        Will render the policy passed in in an infinite loop. You can send a keyboard interrupt (Ctrl-C) and it will
        end the render loop without ending your interactive session.

        Tries to close the window when you send the interrupt, doesn't actually work for Mujoco environments though.

        Attributes:
            policy: your (presumably neural network) function that maps states->actions
            env: the environment you want to render actions in
            select_action: function for actually picking an action from the policy, should be the same one you trained with

        Returns:
            Nothing

        Example:
            import torch
            import torch.nn as nn
            from torch.distributions import Normal
            import gym
            from utils.nn_utils import policy_render_loop

            import os
            print(os.getcwd())

            policy = nn.Sequential(
                nn.Linear(4, 12),
                nn.LeakyReLU(),
                nn.Linear(12, 12),
                nn.LeakyReLU(),
                nn.Linear(12, 1),
            )

            load_path = '/Users/sgillen/work_dir/ucsb/notebooks/rl/cont_working'
            policy.load_state_dict(torch.load(load_path))


            def select_action(policy, state):
                # loc is the mean, scale is the variance
                m = Normal(loc = policy(torch.tensor(state))[0], scale = torch.tensor(.7))
                action = m.sample()
                logprob = m.log_prob(action)
                return action.detach().numpy(), logprob


            env_name = 'InvertedPendulum-v2'
            env = gym.make(env_name)


            policy_render_loop(policy,env,select_action)

            # Blocks until you give a Ctrl-C, then drops you back into the shell


    """

    try:
        state = env.reset()
        while True:
            action, _ = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            env.render()

            if done:
                state = env.reset()

    except KeyboardInterrupt:
        env.close()


class MLP(nn.Module):
    """
    Policy designed to be used with seaguls rl module.
    Simple MLP that has a linear layer at the output
    """

    def __init__(
            self, input_size, output_size, num_layers, layer_size, activation=nn.ReLU, output_activation=nn.Identity, device="cpu"
    ):
        """
         :param input_size: how many inputs
         :param output_size: how many outputs
         :param num_layers: how many HIDDEN layers
         :param layer_size: how big each hidden layer should be
         :param activation: which activation function to use
         """
        super(MLP, self).__init__()

        self.activation = activation()
        self.output_activation = output_activation()

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend([nn.Linear(layer_size, layer_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(layer_size, output_size)

        self.state_means = torch.zeros(input_size,device=device)
        self.state_var = torch.ones(input_size,device=device)

    def forward(self, data):

#        data = (torch.as_tensor(data) - self.state_means) / torch.sqrt(self.state_var)

        for layer in self.layers:
            data = self.activation(layer(data))

        return self.output_activation(self.output_layer(data))



def gaus(x, mu, sig):
    '''
    Implementation of gaussian function.
    Input: 
        - x: tensor of size input_size x number of hidden neurons
        - mu: tensor of size of hidden neurons (mu = expected values/centers)
        - sig: tensor of size of hidden neurons (sigma^2 = variances)
    Output: 
        - tensor of size input_size x number of hidden neurons
    '''
    # if(len(x.shape) == 1): # no batching 
    #     x = torch.unsqueeze(x, 0)
    # out = torch.tensor(np.zeros((len(x), len(mu))).tolist())
    # factor = 1./(torch.sqrt(torch.tensor(2.*np.pi))*sig)
    # out = 1./factor *  torch.exp(-torch.pow((x-mu)/sig,2)/2)
    out = 1./(torch.sqrt(torch.tensor(2.*np.pi))*sig) *  torch.exp(-torch.pow((x-mu)/sig,2)/2)
    # for i_x in range(len(x)):
    #     out[i_x] = 1./factor * torch.exp(-torch.pow((x[i_x]-mu)/sig,2)/2)
    return out

class gaussian(nn.Module):
    '''
    Implementation of gaussian activation function.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - mu - trainable parameter (expected value)
        - sig - trainable parameter (sigma^2 = variance)
    '''
    def __init__(self, hidden_size, mu = None, sig = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - mu, sig: trainable parameter
        '''
        super(gaussian,self).__init__()
        self.bias = False
        # initialize mu and sig
        if mu == None:
            self.mu = nn.Parameter(torch.randn(hidden_size))
            # self.mu = torch.randn(hidden_size)
        else:
            self.mu = nn.Parameter(torch.tensor(mu)) 

        if sig == None:
            self.sig = nn.Parameter(torch.ones(hidden_size))
            # self.sig = torch.ones(hidden_size)
        else:
            self.sig = nn.Parameter(torch.tensor(sig)) 
            
        self.mu.requiresGrad = True # set requiresGrad to true!
        self.sig.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        return gaus(x, self.mu, self.sig)


class RBF(nn.Module):
    """
    Policy designed to be used with seaguls rl module.
    Simple RBF that has one hidden layer with gaussian activation functions at the hidden layer,
    Possible trainable parameter are 
        - weights from input to hidden layer (if input_weights = True)
        - weights from hidden to output layer
        - biases at hidden layer (if input_bias = True)
        - biases at output layer
        - centers of the gaussian activation functions (not implemented yet)
        - variances of the gaussian activation function (not implemented yet)
    """

    def __init__(self, input_size, output_size, layer_size, activation=gaussian, output_activation=nn.Identity):
        """
         :param input_size: how many inputs
         :param output_size: how many outputs
         :param layer_size: how big each hidden layer should be
         :param activation: which activation function to use
         """
        super(RBF, self).__init__()
        self.layer_size = layer_size
        self.activation = activation(layer_size)
        self.output_activation = output_activation()

        self.hidden_layer = nn.Linear(input_size, layer_size)
        input_weights = True
        if(input_weights == False):
            self.hidden_layer.weight = nn.Parameter(torch.ones(layer_size,input_size))
            for p in self.hidden_layer.parameters():
                p.requires_grad = False
        input_bias = False
        if(input_bias == False):
            self.hidden_layer.bias = None
        self.output_layer = nn.Linear(layer_size, output_size)

        self.state_means = torch.zeros(input_size)
        self.state_var = torch.ones(input_size)
        

    def forward(self, data):
        
        data = (torch.as_tensor(data) - self.state_means)/torch.sqrt(self.state_var)

        data = self.activation(self.hidden_layer(data))

        return self.output_activation(self.output_layer(data))



class CategoricalMLP(nn.Module):
    """
    Policy designed to be used with seaguls rl module.
    Simple MLP that will output class label probs
    """

    def __init__(self, input_size, output_size, num_layers, layer_size, activation):
        """
        :param input_size: how many inputs
        :param output_size: how many outputs
        :param num_layers: how many HIDDEN layers
        :param layer_size: how big each hidden layer should be
        :param activation: which activation function to use
        """
        super(CategoricalMLP, self).__init__()

        self.activation = activation()

        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
        self.layers.extend([nn.Linear(layer_size, layer_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(layer_size, output_size)

        if output_size == 1:
            self.output_norm = nn.Sigmoid()
        else:
            self.output_norm = nn.Softmax(dim=-1)

        self.state_means = torch.zeros(input_size)
        self.state_var = torch.ones(input_size)

    def forward(self, data):
        data = (data - self.state_means) / torch.sqrt(self.state_var)

        for layer in self.layers:
            data = self.activation(layer(data))

        return self.output_norm(self.output_layer(data))


class DummyNet(nn.Module):
    """
    This is a dummy network used for debugging, it can be used as a drop in replacement for 
    any network but you can set the net_fn member to whatever function you want and this net 
    will always return. The default function always returns zeros

    example:
        import torch.nn as nn
        from seagul.nn import CategoricalMLP, MLP, DummyNet
        import torch

        from seagul.sims.cartpole import LQRControl
        torch.set_default_dtype(torch.double)

        policy = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
        value_fn = MLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)

    #    gate_fn = CategoricalMLP(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
        gate_fn = DummyNet(input_size=4, output_size=1, layer_size=12, num_layers=2, activation=nn.ReLU)
        gate_fn.net_fn = lambda x : 1
                    
                       
    
        env_name = "su_cartpole_push-v0"
        env = gym.make(env_name)

        model = SwitchedPPOModel(policy, LQRControl, value_fn, gate_fn, env=env)

        t_model, rewards, arg_dict = ppo_switch(
            env_name, 500, model,  epoch_batch_size=10, action_var_schedule=[10,0], gate_var_schedule=[1,0]
        ) 
    
    """

    def __init__(self, input_size, output_size, num_layers, layer_size, activation):
        """
        :param input_size: how many inputs
        :param output_size: how many outputs
        :param num_layers: ignored
        :param layer_size: ingnored
        :param activation: ignored
        """
        super(DummyNet, self).__init__()
        self.output_size = output_size
        self.layer = nn.Linear(input_size, output_size, bias=False)

    def net_fn(self, data):
        return torch.zeros(self.output_size)

    def forward(self, data):
        dummy = self.layer(data) * torch.zeros(self.output_size)  # so that torch sees a gradient
        return dummy + self.net_fn(data)


class LinearNet(nn.Module):
    """
    This is a "network" consisting of a single linear layer
    """

    def __init__(self, input_size, output_size, bias=False):
        """
        :param input_size: how many inputs
        :param output_size: how many outputs
        """
        super(LinearNet, self).__init__()
        self.output_size = output_size
        self.layer = nn.Linear(input_size, output_size, bias=bias)  #

        self.state_means = torch.zeros(input_size)
        self.state_var = torch.ones(input_size)

    def forward(self, data):
        data = (data - self.state_means) / torch.sqrt(self.state_var)
        return self.layer(data)


# Marco Molnar
def make_histories(states, history_length, sampling_sparsity=1):
    """Used to make a dataset suitable for a neural network who's state is a history of inputs.


    This function takes numpy array states which should be a time series, and returns an array of histories of size
    (history_length, num_states) the optional sampling_sparsity parameter decides how many time steps to look back for
    every entry in a history. This is probably best explained by looking at the return value:

    histories[i] = np.array([states[i], states[i - T], states[i - 2*T], ... ])

    Attributes:
        states:  input numpy array, must be 2 dimensional (num_samples, num_states)
        history_length:  number of samples in each history
        sampling_sparsity:  timesteps between samples in each history

    Returns:
        histories: numpy array (num_samples, num_states, history_length)

    Example:
        states = np.random.randn(12,2)
        history_state = make_histories(states, 3)


      """

    num_set = states.shape[0]
    z_ext = np.zeros(((history_length - 1) * sampling_sparsity, states.shape[1]))
    states = np.concatenate((z_ext, states), axis=0)
    histories = np.zeros((num_set,) + (states.shape[1],) + (history_length,))  # initialize output matrix
    step = 0

    while step < num_set:
        # select vectors according to history_length and sampling_sparsity
        histories[step, :, :] = np.transpose(
            states[step : (history_length - 1) * sampling_sparsity + 1 + step : sampling_sparsity, :]
        )
        step += 1
    return histories


# One day this might be a unit test
if __name__ == "__main__":
    policy = MLP(input_size=4, output_size=1, num_layers=3, layer_size=12, activation=nn.ReLU)

    policy.state_means = torch.ones(4)
    policy.state_var = torch.ones(4) * 4
    print(policy(torch.randn(1, 4)))

    policy = LinearNet(input_size=4, output_size=1, bias=False)
    print(policy(torch.randn(1, 4)))
