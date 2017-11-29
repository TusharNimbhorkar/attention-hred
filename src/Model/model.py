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
from src.Model import encoder_grucell
from src.Model import decoder_grucell


class HERED():
    """"
    This Class includes the methods to build de graph for the Recurrent
      Encoder-Decoder.
    """

    def __init__(self, vocab_size=50004, embedding_dim=300, query_dim=1000, session_dim=1500,
                 decoder_dim=1000, output_dim=50004, unk_symbol=0, eoq_symbol=1, eos_symbol=2,learning_rate=1e-1, hidden_layer=1):
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
        self.hidden_layers = hidden_layer



        # create objects for query encoder, session encoder and decoder.
        # raise NotImplementedError
    def initialise(self, X):

        #Create the embeddings
        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        #Create the query encoder state
        states = encoder_grucell.compute_state(x=embedder)
        self.initial_query_state = encoder_grucell.get_final_state(x=embedder, states=states)
        #Create the session state
        self.initial_session_state = encoder_grucell.compute_state(x=self.initial_query_state)

        self.decoder_state = decoder_grucell.compute_state(x=self.initial_query_state,
                                                      session_state=self.initial_session_state)


    def inference(self, X, Y):

        # call here tf.scan for each.
        # see if we should add an additional output layer after decoder.

        num_of_steps = tf.shape(X)[0]
        batch_size = tf.shape(X)[1]

        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        # Create the query encoder state
        states = encoder_grucell.compute_state(x=embedder)
        self.initial_query_state = encoder_grucell.get_final_state(x=embedder, states=states)
        # Create the session state
        self.initial_session_state = encoder_grucell.compute_state(x=self.initial_query_state)
        #TODO fix this when the decoder is finished
        self.decoder_state = decoder_grucell.compute_state(x=Y,
                                                      session_state=self.initial_session_state)

        logits = layers.output_layer(embedding_dims=self.embedding_dim, vocabulary_size= self.vocab_size, num_hidden= self.hidden_layers,
                                     state=self.decoder_state, word= Y)


        # Calculate the omega function w(d_n-1, w_n-1).
        #  word is the previous word and state the previous hidden state of the decoder
        w = layers.output_layer(self.embedding_dim, self.decoder_dim, self.vocab_size, state, word)

        return logits

    def train_step(self):
        # here it would go the optimizer for the model. However, since it is now 3 RNN
        # and not all of them are optimize or might be done differently maybe this is
        # not needed anymore
        raise NotImplementedError

    def get_loss(self, logits, labels):
        # same as for train_step.....


        labels = tf.one_hot(labels, self.vocab_size)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        tf.summary.scalar('LOSS', loss)
        return loss

    def softmax(self, logits):

        return tf.nn.softmax(logits)

    def optimizer(self,loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def accuracy(self,logits,labels):

        # todo: find out how to calculate accuracy and implement
        accuracy=0

        tf.summary.scalar('Accuracy', accuracy)
        raise NotImplementedError





