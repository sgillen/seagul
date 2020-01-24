import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class RBFModel(TFModelV2):
    def __init__(self, *args, **kwargs):
        super(RBFModel, self).__init__(*args, **kwargs)
        input_layer = tf.keras.layers.Input(3,)
        hidden_layer = RBFLayer(64, 0)(input_layer)
        output_layer = tf.keras.layers.Dense(1)(hidden_layer)
        value_layer = tf.keras.layers.Dense(1)(hidden_layer)
        self.base_model = tf.keras.Model(
            input_layer, [output_layer, value_layer])
        self.register_variables(self.base_model.variables)
    # def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # super(RBFModel, self).__init__(*args, **kwargs)
        # self.hidden_layer = tf.keras.layers.Dense(64, activation = 'relu')
        # self.output_layer = tf.keras.layers.Dense(1, activation = 'linear')

        # self.base_model = tf.keras.models.Sequential([
        #     input_layer = tf.keras.layers.Input()
        #     hidden_layer = tf.keras.layers.RBFLayer(64)
        #     output_layer = tf.keras.layers.Dense(1, activation = 'linear')
        # ])
    def forward(self, input_dict, state, seq_lens):
        x = self.hidden_layer(state)
        # model_out, self._value_out = self.base_model(input_dict["obs"])
        # return model_out, state
        return self.output_layer(x), state
    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model("rbf_model", RBFModel)

ray.init(local_mode=True) # local mode for debugging
tune.run(
    "SAC",
    stop={"episode_reward_mean": -200},
    config={
        "model": {
            "custom_model": "rbf_model",
            "custom_options": {},  # extra options to pass to your model
        },
        "env": "Pendulum-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "eager": False,
    },
)