"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import tensorflow as tf
import numpy as np

class Decoder(object):

    def __init__(self,batch_size, reuse, input_dim=300, num_hidden_query=1000, num_hidden_session=1500):
        """
        This Class implements a Decoder.

        :param batch_size: int, length of batch
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden_query: int, dimensions for the hidden state of the decoder, same as the query_encoder
        :param num_hidden_session: int, dimensions for the hidden state of the session enconder
        """
        self.input_dim = input_dim
        self.num_hidden_query = num_hidden_query
        self.num_hidden_session = num_hidden_session
        self.reuse = reuse
        self.batch_size = batch_size

        initializer_weights = tf.variance_scaling_initializer() #xavier
        initializer_biases = tf.constant_initializer(0.0)

        # Define network architecture
        with tf.variable_scope('Encoder_GRU', reuse = self.reuse):

            # Weights for reset gate
            self.Hr = tf.get_variable(shape=[self.num_hidden_query, self.num_hidden_query], initializer=initializer_weights,
                                       name='Hr')
            self.Ir = tf.get_variable(shape=[self.num_hidden_query, self.input_dim], initializer=initializer_weights,
                                      name='Ir')
            # Weights for update gate
            self.Hu = tf.get_variable(shape=[self.num_hidden_query, self.num_hidden_query], initializer=initializer_weights,
                                      name='Hu')
            self.Iu = tf.get_variable(shape=[self.num_hidden_query, self.input_dim], initializer=initializer_weights,
                                      name='Iu')
            # Weights for Candidate
            self.Ic = tf.get_variable(shape=[self.num_hidden_query, self.num_hidden_query], initializer=initializer_weights,
                                      name='Ic')
            self.Hc = tf.get_variable(shape=[self.num_hidden_query, self.input_dim], initializer=initializer_weights,
                                      name='Hc')
            # Weights for initial recurrent state
            self.Bo = tf.get_variable(shape= [self.num_hidden_query], initializer= initializer_biases, name= 'Bo')
            self.Do = tf.get_variable(shape= [self.num_hidden_query, self.num_hidden_session], initializer= initializer_biases, name= 'Do' )

    def _gru_step(self, h_prev, x):

        """
        Custom function to implement a recurrent step. To use with tf.scan
        :param h_prev: previous hidden state.
        :param x: data for current timestep
        :return: the next state.
        """
        x, session_reset_vector = tf.unstack(x)

        # Calculate reset gate
        r = tf.sigmoid(tf.matmul(x,self.Ir)+tf.matmul(h_prev, self.Hr))
        # Calculate update gate
        u = tf.sigmoid(tf.matmul(x, self.Iu) + tf.matmul(h_prev, self.Hu))
        # Calculate candidate
        c = tf.tanh(tf.matmul(x, self.Ic) + tf.matmul(r*h_prev, self.Hc) )

        h = tf.subtract(np.float32(1.0),u) * h_prev + u * c


        return h_prev * session_reset_vector + tf.subtract(tf.constant(1.0, tf.float32), session_reset_vector) * h

    def compute_state(self, x, session_state):
        """
        :session_state: state to initialise the recurrent state of the decoder
        :x: array of embeddings of query batch
        :return: query representation tensor
        """
        # Initialise recurrent state with session_state
        init_state = tf.tanh(tf.sum(tf.matmul(session_state, self.Do), self.Bo))
        # Calculate RNN states
        states = tf.scan(self._gru_step, tf.transpose(x), initializer= init_state)
        return states[-1]