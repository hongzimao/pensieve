import numpy as np
import tensorflow as tf
import tflearn


class PredictionNetwork(object):

    def __init__(self, sess, s_dim, a_dim, learning_rate):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_rate = learning_rate

        # Create neural network
        self.inputs, self.out = self.create_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target
        self.target = tf.placeholder(tf.float32, [None, self.a_dim])

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.out))

        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
                        minimize(self.cross_entropy, var_list=self.network_params)

    def create_network(self):
        with tf.variable_scope('nn'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :], 128, 4, activation='relu')

            merge_net = tflearn.merge([split_0, split_1, split_2, split_3, split_4], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 100, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='linear')

            return inputs, out

    def train(self, inputs, target):
        return self.sess.run([self.cross_entropy, self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.target: target
            })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def build_summaries():

    loss = tf.Variable(0.)
    tf.scalar_summary("Loss", loss)
    
    summary_vars = [loss]
    summary_ops = tf.merge_all_summaries()

    return summary_ops, summary_vars