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




class RBFLayer(Layer):
    def __init__(self, config, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = config["units"]
        self.normalization = config["normalization"]
        self.same_smooth_fac = config["const_beta"]
        self.beta_initial = config["beta_initial"]
    def build(self, input_shape):
        initializer_gaus = RandomNormal(mean=0.0, stddev=1.0, seed=None)
        self.mu = self.add_weight(name='mu',
                                  shape=(input_shape[1], self.units, 1), # centers have the same dimension as the input data (x number of neurons)
                                  initializer=initializer_gaus,
                                  dtype="float32",
                                  trainable=True)
        if self.same_smooth_fac:
            self.beta = self.add_weight(name='beta',
                                  shape=(self.units,1),
                                  initializer=self.beta_initial,
                                  dtype="float32",
                                  trainable=False)
        else:
            self.beta = self.add_weight(name='beta',
                                  shape=(self.units,1),
                                  initializer=self.beta_initial,
                                  dtype="float32",
                                  trainable=True)
        
        # self.input_weights = self.add_weight(name='input_weights',
        #                           shape = (self.units,),
        #                           initializer = initializer_gaus,
        #                           trainable = True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        # TODO: implement mahalanobis distance to compare results
        inputs = tf.dtypes.cast(inputs, tf.float32)
        # import ipdb; ipdb.set_trace()
        inputs = K.transpose(K.expand_dims(inputs,1))
        # with norm:
        # rho = K.exp(- self.beta * K.pow(tf.norm(inputs - self.mu, ord = 'euclidean', axis = 0),2))
        # same as with norm but less comp cost:
        if self.same_smooth_fac:
            rho = K.exp(- self.beta * K.pow(K.sum(inputs - self.mu, axis = 0),2))
        else:
            rho = K.exp(- tf.math.abs(self.beta) * K.pow(K.sum(inputs - self.mu, axis = 0),2)) # beta has to be positive
        if self.normalization:
            return K.transpose(rho / K.sum(rho, axis = 0))
        else:
            return K.transpose(rho)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class RBFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(RBFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        initializer_gaus = RandomNormal(mean=0.0, stddev=1.0, seed=None)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations",
            dtype=tf.float32)
        hidden_layer = RBFLayer(
            model_config["custom_options"])(self.inputs)
        output_layer = tf.keras.layers.Dense(
            num_outputs,
            name="my_output_layer",
            activation=None,
            use_bias = True,
            kernel_initializer=initializer_gaus)(hidden_layer)
        value_layer = tf.keras.layers.Dense(
            1,
            name="my_value_layer",
            activation=None,
            use_bias = True,
            kernel_initializer=initializer_gaus)(hidden_layer)
        self.base_model = tf.keras.Model(
            self.inputs, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        self._value_out = K.squeeze(self._value_out, axis=-1)
        return model_out, state
    def value_function(self):
        return self._value_out
        # return tf.reshape(self._value_out, [-1])