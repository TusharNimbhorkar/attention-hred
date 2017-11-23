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

    def __init__(self,  reset_emb, batch_size, reuse, input_dim=300, num_hidden=1000):

        """

        This Class implements an Encoder.

        :param batch_size: int, length of batch
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden: int, dimensions for the hidden state of the encoder
        """

        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.reuse = reuse
        self.batch_size = batch_size


        initializer_weights = tf.variance_scaling_initializer() #xavier
        #initializer_biases = tf.constant_initializer(0.0)

        # Define network architecture
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
        """
        Custom function to implement a recurrent step. To use with tf.scan
        :param h_prev: previous hidden state.
        :param x: data for current timestep
        :return: the next state.
        """
        x, reset_vector = tf.unstack(x)
        _, h_prev = tf.unstack(h_prev)
        # Calculate reset
        r = tf.sigmoid(tf.matmul(x,self.Ir)+tf.matmul(h_prev, self.Hr))
        # Calculate update
        u = tf.sigmoid(tf.matmul(x, self.Iu) + tf.matmul(h_prev, self.Hu))
        # Calculate candidate
        c = tf.tanh(tf.matmul(x, self.Ic) + tf.matmul(r*h_prev, self.Hc) )

        h = tf.subtract(np.float32(1.0),u) * h_prev + u * c

        """"
         h gives you the hidden state without taking into account the end of query,  h * reset_vector will reset
         the hidden state to zero if its the end of the query, the first parameter will be used by the session encoder,
         while the second parameter is used as the actual hidden state of the encoder so that the encoder resets when there
         is an end of query symbol
        """

        return tf.stack([h, h * reset_vector])

    def compute_state(self, x):
        """
        :x: array of embeddings of query batch and reset vector 1 when embedding is not end of session/query 0 when it is
        :return: query representation tensor
        """
        # Initialise recurrent state
        init_state = tf.zeros([2, self.batch_size, self.num_hidden], name='init_state')
        # Calculate RNN states
        states = tf.scan(self._gru_step, x , initializer= init_state)
        original_states, _ = tf.unstack(states)
        _, reset_vector = tf.unstack(x)

        return ([original_states, reset_vector])

    def compute_state_session(self, x):
        """
        :x: array of embeddings of query batch and reset vector 1 when embedding is not end of session/query 0 when it is
        :return: query representation tensor
        """
        x, reset_vector = tf.unstack(x)
        # Calculate RNN states
        states = tf.scan(self._gru_step,tf.transpose(x), initializer=self.init_state)
        # Update recurrent state if  received entire query
        self.init_state = states[-2] * reset_vector + tf.sub(tf.constant(1.0, tf.float32) , reset_vector)*states[-1]
        return states[-1]