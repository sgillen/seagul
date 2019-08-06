# Mostly copied from here https://github.com/openai/baselines/blob/master/baselines/ppo1/mlp_policy.py

from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
import numpy as np
from baselines.common.distributions import make_pdtype


class SeededMlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    # Gaussian_fixed_var was true initially, having it be false with surely mess up everything...
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=False):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope("vf"):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                    tf.layers.dense(
                        last_out, hid_size, name="fc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)
                    )
                )
            self.vpred = tf.layers.dense(last_out, 1, name="final", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        with tf.variable_scope("pol", reuse=tf.AUTO_REUSE):
            last_out = obz
            # for i in range(num_hid_layers):

            fc1 = tf.layers.dense(obz, hid_size, name="fc1", kernel_initializer=U.normc_initializer(1.0))
            fc1_tanh = tf.nn.tanh(fc1)
            fc2 = tf.layers.dense(fc1_tanh, hid_size, name="fc2", kernel_initializer=U.normc_initializer(1.0))
            fc2_tanh = tf.nn.tanh(fc2)
            final = pdparam = tf.layers.dense(
                fc2_tanh, pdtype.param_shape()[0], name="final", kernel_initializer=U.normc_initializer(0.01)
            )

            # param_dict = np.load('./model_weights.npz')

            # saver = tf.train.import_meta_graph('cartpole_test.meta')
            # tf.Session()
            # print(tf)

        pol_vars = tf.trainable_variables("pi/pol/")
        # var_list = []
        # for var in pol_vars:
        #    separator = "/"
        #    name_without_pi  = separator.join(var.name.split("/")[1:])
        #    var_list.append(tf.get_variable(name_without_pi))

        saver = tf.train.Saver(pol_vars)
        saver.restore(tf.Session(), "./data/supervised_weights")
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

        with tf.Session() as sess:
            writer = tf.summary.FileWriter("ppo_out", sess.graph)
            writer.add_graph(graph=sess.graph)
            writer.close()

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
