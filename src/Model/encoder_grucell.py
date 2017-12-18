"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self, batch_size, level, input_dim=300, num_hidden=1000):

        """

        This Class implements an Encoder.

        :param batch_size: int, length of batch
        :param level: specify wether it's query or session level encoder
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden: int, dimensions for the hidden state of the encoder
        """

        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.level      = level

        initializer_weights = tf.variance_scaling_initializer() #xavier
        initializer_biases = tf.constant_initializer(0.0)
        # https://stackoverflow.com/questions/45456116/valueerror-trying-to-share-variable-enco-gru-cell-gates-kernel-but-specified-s
        with tf.variable_scope('gru_encoder', reuse=tf.AUTO_REUSE):
            self.gru_cell = tf.contrib.rnn.GRUCell(num_hidden, kernel_initializer=initializer_weights,
                                               bias_initializer=initializer_biases)
        #self.zero_state = self.gru_cell.zero_state(batch_size, tf.float32)

    def length(self,sequence):
        """
         :sequence: batch of padded length; with zero vectors after eoq_symbol 
         :return:   vector determining length of queries (at what point the eoq_symbol is encountered)
        """
        used = tf.sign(tf.reduce_max(tf.abs(tf.convert_to_tensor(sequence)), 2))
        # used.shape
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def compute_state(self, x, state=None):
        """
        :x:      query/session batch of padded length [batch_size x max_length x out_size]
        :state:  previous state. used initialize the encoder. if not given it initializes with zero_state
        :return: states representation per query/session step [batch_size x max_length x out_size]
        """
        # Initialise recurrent state
        if not state:
            state = self.gru_cell.zero_state(self.batch_size, tf.float32)

        if self.level == 'query': 
            length = self.length(tf.convert_to_tensor(x))
            # Calculate RNN states
            _, state = tf.nn.dynamic_rnn(
                self.gru_cell,
                x,
                dtype=tf.float32,
                sequence_length=length,
                initial_state=state,
                swap_memory=True,
                scope=self.level)
        elif self.level=='session':
            # hoping this is right and GRUs outputs is equivalent to state (second answer)
            # https://stackoverflow.com/questions/39716241/tensorflow-getting-all-states-from-a-rnn
            state, _ = tf.nn.static_rnn(
                self.gru_cell,
                [x],
                dtype=tf.float32,
                initial_state=state,
                scope=self.level)
        else:
            raise BaseException('Values for Encoder.level can only be "query" or "session"')
        return state

    def get_final_state(self, x, states):
        """
        :x:      query/session batch of padded length [batch_size x max_length x out_size]
        :states: output of compute_state
        :return: vector of all the final states per query
        """
        length = self.length(x)
        batch_size = tf.shape(states)[0]
        max_length = tf.shape(states)[1]
        out_size = tf.shape(states)[2]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(states, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
        #return states[:, self.length(x)-1] # TensorFlow does’t support advanced slicing yet
