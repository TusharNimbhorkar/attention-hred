"""
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

"""

import tensorflow as tf
import numpy as np

class Decoder(object):

    def __init__(self, input_dim=300, num_hidden_query=1000, num_hidden_session=1500):
        """
        This Class implements a Decoder.
        :param reuse: bool, parameter to reuse tensorflow variables
        :param input_dim: int, word embedding dimensions
        :param num_hidden_query: int, dimensions for the hidden state of the decoder, same as the query_encoder
        :param num_hidden_session: int, dimensions for the hidden state of the session enconder
        """
        self.input_dim = input_dim
        self.num_hidden_query = num_hidden_query
        self.num_hidden_session = num_hidden_session

        initializer_weights = tf.variance_scaling_initializer() #xavier
        initializer_biases = tf.constant_initializer(0.0)

        self.gru_cell = tf.contrib.rnn.GRUCell(num_hidden_query, kernel_initializer=initializer_weights, bias_initializer=initializer_biases)


    def length(self, sequence):
        """
         :sequence: batch of padded length; with zero vectors after eoq_symbol 
         :return:   vector determining length of queries (at what point the eoq_symbol is encountered)
        """
        used = tf.sign(tf.reduce_max(tf.abs(tf.convert_to_tensor(sequence)), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def compute_state(self, x, session_state, query_encoder_last_state=None):
        """
        :x:             query/session batch of padded length [batch_size x in_size]
        :session_state: state to initialise the recurrent state of the decoder
        :return:        query representation tensor [batch_size x in_size]
        """
        # Initialise recurrent state with session_state
        # state = tf.tanh(tf.matmul(tf.squeeze(session_state), tf.transpose(self.Do)) + self.Bo)
        # _, state = self.gru_cell(query_encoder_last_state, state)

        # TODO: check that the x is right because maybe the decoder is not reusing its outputs

        # Calculate RNN states
        # length = self.length(tf.convert_to_tensor(x))
        _, states = tf.nn.static_rnn(
            self.gru_cell,
            [x],
            dtype=tf.float32,
            initial_state= session_state)
        return states

    def compute_prediction_using_while_loop(self, first_state, query_encoder_last_state, sequence_length):
        """
        :session_state:            state to initialize the recurrent state of the decoder
        :query_encoder_last_state: last encoder state of the previous query to be used as first input
        """
        output, state = self.gru_cell(query_encoder_last_state, first_state)
        outputs = tf.expand_dims(output, 2)
        states = tf.expand_dims(state, 2)
        # Calculate RNN states
        # stop_after = sequence_length

        c = lambda o,s,os,ss,i: tf.greater(i, 0)
        b = lambda o,s,os,ss,i: self.concat_fn(o,s,os,ss,i)

        _, _, outputs, states, _ = tf.while_loop(cond=c, body=b,loop_vars=(output,state,outputs,states,sequence_length),
                                                 name='while',shape_invariants=(tf.TensorShape([output.get_shape()[0], output.get_shape()[1]]),
                                                                                tf.TensorShape([state.get_shape()[0], state.get_shape()[1]]),
                                                                                tf.TensorShape([outputs.get_shape()[0], outputs.get_shape()[1], None]),
                                                                                tf.TensorShape([states.get_shape()[0], states.get_shape()[1], None]),
                                                                                tf.TensorShape(None)))
        return outputs, states

    def compute_prediction(self, y, state, batch_size, vocab_size):
        """
        :session_state:            state to initialize the recurrent state of the decoder
        :query_encoder_last_state: last encoder state of the previous query to be used as first input
        """
        # Add first input (zeros)
        print('here')
        print(y)
        y_one_hot_shifted = tf.one_hot(y, depth=vocab_size)
        print(y_one_hot_shifted)
        start_word        = tf.expand_dims(tf.zeros([batch_size, vocab_size]), 1)
        y_one_hot_shifted= tf.concat([start_word, y_one_hot_shifted], 1)[:,:-1]

        length = self.length(tf.convert_to_tensor(y_one_hot_shifted))+1
        # Calculate RNN states
        outputs, _ = tf.nn.dynamic_rnn(
            self.gru_cell,
            y_one_hot_shifted,
            dtype=tf.float32,
            sequence_length=length,
            initial_state=state)
        return outputs

    def concat_fn(self,output,state,outputs,states,seq_len):
        output, state = self.gru_cell(output, state)
        outputs = tf.concat([outputs, tf.expand_dims(output, 2)], 2)
        states  = tf.concat([states, tf.expand_dims(state, 2)], 2)
        seq_len = tf.subtract(seq_len, 1)
        return output,state,outputs,states,seq_len

    def check_if_eoq(self, gru_output):
        # Check if the output is the end of query symbol
        raise NotImplementedError

    def calculate_word(self, gru_output):
        # Calculate word given the output of the Decoder 
        raise NotImplementedError

    def prediction(self, gru_output):
        # Transform output to prediction given the output of the Decoder
        raise NotImplementedError
