import numpy as np
import tensorflow as tf
import tflearn


GAMMA = 0.99
ENTROPY_WEIGHT = 0.1
ENTROPY_EPS = 1e-6
EPS = 1e-6
MAX_BR_LEVELS = 10
MASK_DIM = 6


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        assert self.a_dim == MAX_BR_LEVELS

        # Placeholder for masking invalid actions
        self.mask = tf.placeholder(tf.bool, self.a_dim)

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        # the shape of acts are not determined (only upper-bounded by a_dim)
        self.acts = tf.placeholder(tf.float32, [None, None])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.mul(
                       tf.log(tf.reduce_sum(tf.mul(self.out, self.acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.mul(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 64, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            split_2 = tflearn.fully_connected(inputs[:, 4:5, -1], 64, activation='relu')

            reshape_0 = tflearn.reshape(inputs[:, 2:4, :], [-1, 2, self.s_dim[1], 1])
            split_3 = tflearn.conv_2d(reshape_0, 128, 3, activation='relu')

            split_4 = tflearn.conv_1d(inputs[:, 5:6, :], 128, 4, activation='relu')
            split_5 = tflearn.conv_1d(inputs[:, 6:7, :], 128, 4, activation='relu')

            flatten_0 = tflearn.flatten(split_3)
            flatten_1 = tflearn.flatten(split_4)
            flatten_2 = tflearn.flatten(split_5)

            merge_net = tflearn.merge([split_0, split_1, split_2, flatten_0, flatten_1, flatten_2], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')

            # for multiple video, mask out the invalid actions
            linear_out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='linear')
            linear_out = tf.transpose(linear_out)  # [None, a_dim] -> [a_dim, None]
            mask_out = tf.boolean_mask(linear_out, self.mask)  # [a_dim, None] -> [masked, None]
            mask_out = tf.transpose(mask_out)  # [masked, None] -> [None, masked]
            softmax_out = tf.nn.softmax(mask_out)

            return inputs, softmax_out

    def train(self, inputs, acts, act_grad_weights):
        # there can be only one kind of mask in a training epoch
        for i in xrange(inputs.shape[0]):
            assert np.all(inputs[0, MASK_DIM, -MAX_BR_LEVELS:] == \
                          inputs[i, MASK_DIM, -MAX_BR_LEVELS:])

        # action dimension matches with mask length
        assert acts.shape[1] == np.sum(inputs[0:1, MASK_DIM, -MAX_BR_LEVELS:])

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.mask: inputs[0, MASK_DIM, -MAX_BR_LEVELS:],
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        for i in xrange(inputs.shape[0]):
            assert np.all(inputs[0, MASK_DIM, -MAX_BR_LEVELS:] == \
                          inputs[i, MASK_DIM, -MAX_BR_LEVELS:])

        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.mask: inputs[0, MASK_DIM, -MAX_BR_LEVELS:]
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        for i in xrange(inputs.shape[0]):
            assert np.all(inputs[0, MASK_DIM, -MAX_BR_LEVELS:] == \
                          inputs[i, MASK_DIM, -MAX_BR_LEVELS:])

        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.mask: inputs[0, MASK_DIM, -MAX_BR_LEVELS:],
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.sub(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 64, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 64, activation='relu')
            split_2 = tflearn.fully_connected(inputs[:, 4:5, -1], 64, activation='relu')

            reshape_0 = tflearn.reshape(inputs[:, 2:4, :], [-1, 2, self.s_dim[1], 1])
            split_3 = tflearn.conv_2d(reshape_0, 128, 3, activation='relu')

            split_4 = tflearn.conv_1d(inputs[:, 5:6, :], 128, 4, activation='relu')
            split_5 = tflearn.conv_1d(inputs[:, 6:7, :], 128, 4, activation='relu')

            flatten_0 = tflearn.flatten(split_3)
            flatten_1 = tflearn.flatten(split_4)
            flatten_2 = tflearn.flatten(split_5)

            merge_net = tflearn.merge([split_0, split_1, split_2, flatten_0, flatten_1, flatten_2], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 100, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(xrange(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in xrange(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.scalar_summary("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.scalar_summary("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.scalar_summary("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars
