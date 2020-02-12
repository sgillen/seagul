import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.layers import Layer
from keras.initializers import RandomNormal, RandomUniform, glorot_uniform
from tensorflow.keras import backend as K
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.visionnet_v2 import VisionNetwork as MyVisionNetwork
import datetime
from gym import envs


class MLP(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MLP, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.hidden_neurons = model_config["custom_options"]["hidden_neuons"]
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            self.hidden_neurons[0],
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        if len(self.hidden_neurons) == 2:
            layer_2 = tf.keras.layers.Dense(
                self.hidden_neurons[1],
                name="my_layer2",
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))(layer_1)
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="my_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_2)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_2)
        else:
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="my_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_1)
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class Linear(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(Linear, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(self.inputs)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(self.inputs)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])