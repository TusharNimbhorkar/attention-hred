"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

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
        initializer_biases = tf.constant_initializer(0.0)
        self.gru_cell = tf.contrib.rnn.GRUCell(num_hidden, kernel_initializer=initializer_weights, bias_initializer=initializer_biases)
        #self.zero_state = self.gru_cell.zero_state(batch_size, tf.float32)

    def length(sequence):
        """
         :sequence: batch of padded length; with zero vectors after eoq_symbol 
         :return:   vector determining length of queries (at what point the eoq_symbol is encountered)
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def compute_state(self, x, batch_size, state=None):
        """
        :x:      query/session batch of padded length [batch_size x max_length x out_size]
        :state:  previous state. used to not initialize with zero_state
        :return: states representation per query/session step [batch_size x max_length x out_size]
        """
        # Initialise recurrent state
        if not state:
            state = self.gru_cell.zero_state(batch_size, tf.float32)

        # Calculate RNN states
        _, states = tf.nn.dynamic_rnn(
            self.gru_cell,
            x,
            dtype=tf.float32,
            sequence_length=self.length(x),
            initial_state=state)

        return states

    def get_final_state(self, x, states):
        """
        :x:      query/session batch of padded length [batch_size x max_length x out_size]
        :states: output of compute_state
        :return: vector of all the final states per query
        """
        return states[:, self.length(x)-1]
