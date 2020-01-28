import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow.keras.layers import Layer
from keras.initializers import RandomNormal
from tensorflow.keras import backend as K
import tensorflow as tf
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.visionnet_v2 import VisionNetwork as MyVisionNetwork
import datetime
import gym


class RBFLayer(Layer):
    def __init__(self, units, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        initializer_gaus = RandomNormal(mean=0.0, stddev=1.0, seed=None)
        self.use_weights = True
        self.mu = self.add_weight(name='mu',
                                  shape=(self.units, input_shape[1]), # centers have the same dimension as the input data (x number of neurons)
                                  initializer=initializer_gaus,
                                  dtype="float32",
                                  trainable=True)
        self.beta = self.add_weight(name='beta',
                                  shape=(self.units,),
                                  initializer='ones',
                                  dtype="float32",
                                  trainable=True)
        self.input_weights = self.add_weight(name='input_weights',
                                  shape = (self.units,),
                                  initializer = 'random_uniform',
                                  trainable = True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.dtypes.cast(inputs, tf.float32)
        mu_new = K.expand_dims(self.mu)  
        dist = K.sum((K.transpose(mu_new-K.transpose(inputs)))**2, axis=1) # 2 norm 
        rbf_normalization = K.sum(K.exp(-dist), axis=1) # sum_i(exp(-(x-c_i)^2))
        rbf_normalization = K.expand_dims(rbf_normalization)
        rbf = K.exp(-self.beta * dist) # radial basis function (simplified gaussian with beta as width of RBF)
        nrbf = rbf/rbf_normalization # normalized rbf
        if self.use_weights == True:
            return self.input_weights * nrbf
        else:
            return nrbf

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class RBFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(RBFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations",
            dtype=tf.float32)
        hidden_layer = RBFLayer(
            256)(self.inputs)
        output_layer = tf.keras.layers.Dense(
            num_outputs,
            name="my_output_layer",
            activation=None,
            use_bias = True,
            kernel_initializer='random_uniform')(hidden_layer)
        value_layer = tf.keras.layers.Dense(
            1,
            name="my_value_layer",
            activation=None,
            use_bias = True,
            kernel_initializer='random_uniform')(hidden_layer)
        self.base_model = tf.keras.Model(
            self.inputs, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
    def value_function(self):
        return self._value_out
        # return tf.reshape(self._value_out, [-1])

class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
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