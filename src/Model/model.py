'''
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

'''
import tensorflow as tf
import numpy as np
import layers
# import encoder
# import decoder
from encoder_grucell import Encoder
from decoder_grucell import Decoder


class HERED():
    """"
    This Class includes the methods to build de graph for the Recurrent
      Encoder-Decoder.
    """

    def __init__(self, vocab_size=50004, embedding_dim=300, query_dim=1000, session_dim=1500,
                 decoder_dim=1000, output_dim=50004, unk_symbol=0, eoq_symbol=1, eos_symbol=2,learning_rate=1e-1, hidden_layer=1,batch_size = 50):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
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
        self.query_encoder = Encoder(batch_size = self.batch_size, level='query')
        self.session_encoder = Encoder(batch_size = self.batch_size, level='session', input_dim=1000, num_hidden=1500)
        self.decoder_grucell = Decoder()



        # create objects for query encoder, session encoder and decoder.
        # raise NotImplementedError
    def initialise(self, X):

        #Create the embeddings
        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        #Create the query encoder state
        states = self.query_encoder.compute_state(x=embedder)
        #Create the session state
        self.initial_session_state = self.session_encoder.compute_state(x=self.initial_query_state)

        self.decoder_state = self.decoder_grucell.compute_state(x=self.initial_query_state,
                                                      session_state=self.initial_session_state)


    def inference(self, X, Y):

        # call here tf.scan for each.
        # see if we should add an additional output layer after decoder.

        num_of_steps = tf.shape(X)[0]
        batch_size = tf.shape(X)[1]

        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        y_embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=Y)
        # Create the query encoder state
        self.initial_query_state = self.query_encoder.compute_state(x=embedder)
        # Create the session state
        self.initial_session_state = self.session_encoder.compute_state(x=self.initial_query_state)
        # todo make variable for 1000 here
        self.initial_decoder_state = layers.decoder_initialise_layer(self.initial_session_state[0], 1000)

        # self.initial_decoder_state.shape()

        self.decoder_state = self.decoder_grucell.compute_state(x=y_embedder,
                                                      session_state=self.initial_decoder_state,
                                                      query_encoder_last_state=self.initial_query_state)


        logits = layers.output_layer(embedding_dims=self.embedding_dim, vocabulary_size= self.vocab_size, num_hidden= 1000,
                                     state=self.decoder_state, word= Y)

        # Calculate the omega function w(d_n-1, w_n-1).
        #  word is the previous word and state the previous hidden state of the decoder
        #w = layers.output_layer(self.embedding_dim, self.decoder_dim, self.vocab_size, state, word)

        return logits

    def get_predictions(self, X):
        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        # Create the query encoder state
        self.initial_query_state = self.query_encoder.compute_state(x=embedder)
        # Create the session state
        self.initial_session_state = self.session_encoder.compute_state(x=self.initial_query_state)
        outputs, self.decoder_state = self.decoder_grucell.compute_prediction(session_state=self.initial_session_state,
                                                                            query_encoder_last_state=self.initial_query_state,
                                                                            sequence_length=10)# todo set sequen_length=max_size
        logits = layers.output_layer(embedding_dims=self.embedding_dim, vocabulary_size= self.vocab_size, num_hidden= self.hidden_layers,
                                     state=self.decoder_state, word=outputs)

        return logits

    def get_loss(self, logits, labels):
        # same as for train_step.....

        labels = tf.one_hot(labels, self.vocab_size)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        tf.summary.scalar('LOSS', loss)
        return loss

    def get_auc(self, labels, predictions):
        """

        :param labels: list of labels in int64
        :param predictions:  list of prediictions size (none, self.vocabsize) in one hot encoding
        :return: the area under the curve of the predictions
        """
        one_hot_labels = tf.one_hot(indices= labels, depth=self.vocab_size)
        return tf.metrics.auc(labels= labels, predictions = predictions)


    def softmax(self, logits):
        return tf.nn.softmax(logits)

    def optimizer(self,loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def accuracy(self,logits,labels):
        # todo: find out how to calculate accuracy and implement
        accuracy=0

        tf.summary.scalar('Accuracy', accuracy)
        return accuracy




