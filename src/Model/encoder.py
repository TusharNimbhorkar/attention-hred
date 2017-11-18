"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

File to implement the methods that describe each layer needed for the model.

"""

import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self,batch_size, reuse, input_dim=300, num_hidden=1000):

        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.reuse = reuse
        self.batch_size = batch_size

        initializer_weights = tf.variance_scaling_initializer() #xavier
        #initializer_biases = tf.constant_initializer(0.0)

        #Define network architecture
        with tf.variable_scope('Encoder_GRU', reuse = self.reuse):

            # Weights for reset gate
            self.Hr = tf.get_variable(shape=[self.num_hidden, self.num_hidden], initializer=initializer_weights,
                                       name='Hr')
            self.Ir = tf.get_variable(shape=[self.num_hidden, self.input_dim], initializer=initializer_weights,
                                      name='Ir')
            # Weights for update gate
            self.Hu = tf.get_variable(shape=[self.num_hidden, self.num_hidden], initializer=initializer_weights,
                                      name='Hu')
            self.Iu = tf.get_variable(shape=[self.num_hidden, self.input_dim], initializer=initializer_weights,
                                      name='Iu')
            # Weights for Candidate
            self.Ic = tf.get_variable(shape=[self.num_hidden, self.num_hidden], initializer=initializer_weights,
                                      name='Ic')
            self.Hc = tf.get_variable(shape=[self.num_hidden, self.input_dim], initializer=initializer_weights,
                                      name='Hc')

    def _gru_step(self, h_prev, x):

        # Calculate reset
        r = tf.sigmoid(tf.matmul(x,self.Ir)+tf.matmul(h_prev, self.Hr))
        # Calculate update
        u = tf.sigmoid(tf.matmul(x, self.Iu) + tf.matmul(h_prev, self.Hu))
        # Calculate candidate
        c = tf.tanh(tf.matmul(x, self.Ic) + tf.matmul(r*h_prev, self.Hc) )

        return tf.subtract(np.float32(1.0),u) * h_prev + u * c

    def compute_state(self, x):
        """
        :x: array of embeddings of query batch
        :return: query representation tensor
        """

        init_state = tf.zeros([self.batch_size, self.num_hidden], name= 'init_state')
        states = tf.scan(self._gru_step, tf.transpose(x), initializer= init_state)
        return states[-1]