'''
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

'''
import tensorflow as tf
import numpy as np
import layers
import encoder
import decoder


class HERED():
    """"
    This Class includes the methods to build de graph for the Recurrent
      Encoder-Decoder.
    """

    def __init__(self, vocab_size=50004, embedding_dim=300, query_dim=1000, session_dim=1500,
                 decoder_dim=1000, output_dim=50004, unk_symbol=0, eoq_symbol=1, eos_symbol=2,learning_rate=1e-1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.query_dim = query_dim
        self.session_dim = session_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.unk_symbol = unk_symbol
        self.eoq_symbol = eoq_symbol
        self.eos_symbol = eos_symbol
        self.learning_rate = learning_rate
        # create objects for query encoder, session encoder and decoder.
        # raise NotImplementedError

    def inference(self, X):

        # call here tf.scan for each.
        # see if we should add an additional output layer after decoder.

        num_of_steps = tf.shape(X)[0]
        batch_size = tf.shape(X)[1]

        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        logits=0#Todo:calculate logits somehow

        return logits

    def train_step(self):
        # here it would go the optimizer for the model. However, since it is now 3 RNN
        # and not all of them are optimize or might be done differently maybe this is
        # not needed anymore
        raise NotImplementedError

    def get_loss(self, X, logits, labels):
        # same as for train_step.....
        labels = tf.one_hot(labels, self.vocab_size)

        eos_mask = tf.expand_dims(tf.cast(tf.not_equal(X, self.eos_symbol), tf.float32), 2)
        labels = labels * eos_mask

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        tf.scalar_summary("LOSS", loss)
        return loss

    def softmax(self, logits):

        return tf.nn.softmax(logits)

    def optimizer(self,loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def accuracy(self,logits,labels):

        # todo: find out how to calculate accuracy and implement


        raise NotImplementedError





