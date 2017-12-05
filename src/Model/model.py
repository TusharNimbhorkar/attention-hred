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


    def initialise(self, X):
        """
        Function to initialise the architecture. It runs until a session state is created.
        :param X: input data batch
        """

        # Create the embeddings
        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X)
        # Create the query encoder state
        states = self.query_encoder.compute_state(x=embedder)
        # Create the session state
        self.initial_session_state = self.session_encoder.compute_state(x=self.initial_query_state)

        self.decoder_state = self.decoder_grucell.compute_state(x=self.initial_query_state,
                                                      session_state=self.initial_session_state)


    def inference(self, X, Y, sequence_max_length):

        """
        Function to run the model.
        :param X: data batch [batch_size x max_seq]
        :param Y: target batch
        :param sequence_max_length: max_seq of the batch
        :return: logits [N, hidden size] where N is the number of words (including eoq) in the batch
        """

        embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                              embedding_dims=self.embedding_dim, data=X,scope='X_embedder')

        # Create the query encoder state
        self.initial_query_state = self.query_encoder.compute_state(x=embedder)  # batch_size x query_dims
        # Create the session state
        self.initial_session_state = self.session_encoder.compute_state(x=self.initial_query_state)  # batch_size x session_dims
        # Create the initial decoder state
        self.initial_decoder_state = layers.decoder_initialise_layer(self.initial_session_state[0], self.decoder_dim)  # batch_size x decoder_dims

        # Run decoder and retrieve outputs and states for all timesteps
        self.decoder_outputs, self.decoder_states = self.decoder_grucell.compute_prediction(  # batch size x timesteps x output_size
            first_state=self.initial_decoder_state,
            query_encoder_last_state=self.initial_query_state,
            sequence_length=sequence_max_length)

        # Remove mask from outputs of decoder
        print(self.decoder_outputs.shape)
        mask = self.decoder_grucell.length(embedder)  # get length for every example in the batch
        dec_out = self.decoder_outputs.get_shape()[2]
        result = tf.slice(self.decoder_outputs, [0, 0, 0], [0, tf.gather(mask, tf.convert_to_tensor(0)), dec_out])
        result = tf.reshape(result, [-1, dec_out])
        for i in range(1, self.batch_size):
            example = tf.slice(self.decoder_outputs, [i, 0, 0],
                               [i, tf.gather(mask, tf.convert_to_tensor(i)), dec_out])
            example = tf.reshape(example, [-1, dec_out])
            result = tf.concat([result, example], 0)

        # Shift y
        y_shifted = tf.concat([tf.zeros(self.batch_size, 1), Y], 1)
        # Calculate the omega function w(d_n-1, w_n-1).
        omega = layers.output_layer(embedding_dims=self.embedding_dim, vocabulary_size= self.vocab_size, num_hidden= self.decoder_dim,
                                     state=self.decoder_state, word=y_shifted) #previous word
        # Get embeddings for decoder output
        y_embedder = layers.get_embedding_layer(vocabulary_size=self.vocab_size,
                                                embedding_dims=self.embedding_dim, data=result, scope='Y_embedder')

        #dot product between omega and embeddings of decoder output
        logits = tf.matmul(omega, y_embedder)

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




