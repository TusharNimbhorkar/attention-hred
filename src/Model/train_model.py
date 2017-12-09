'''
IR2 - Reproduction of "A Hierarchical Recurrent
      Encoder-Decoder for Generative Context-Aware Query Suggestion"
      by Sordoni et al.
Group 8

'''

import tensorflow as tf
import numpy as np
import _pickle as cPickle
import sys
import argparse
import time
from datetime import datetime

# Path to get batch iterator
sys.path.insert(0, '../sordoni/')
import data_iterator
from model import HERED


# todo: put this stuff in arg.parse as well
LEARNING_RATE = 1e-4
BATCH_SIZE = 60
MAX_LENGTH = 50
N_BUCKETS = 20
MAX_STEPS = 10000000
VOCAB_SIZE = 50003
random_seed = 1234
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
EMBEDDING_DIM = 300
QUERY_DIM = 1000
SESSION_DIM = 1500
VOCAB_FILE = '../../data/input_model/train.dict.pkl'
TRAIN_FILE = '../../data/input_model/train.ses.pkl'
VALID_FILE = '../../data/input_model/valid.ses.pkl'


class Train(object):
    def __init__(self,config=None):

        self.config = config
        self.vocab = cPickle.load(open(VOCAB_FILE, 'rb'))
        self.vocab_lookup_dict = {k: v for v, k, count in self.vocab}

        self.train_data, self.valid_data = data_iterator.get_batch_iterator(np.random.RandomState(random_seed), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': config.buckets,
            'bs': config.batch_size,
            'train_session': TRAIN_FILE,
            'seqlen': config.max_length,
            'valid_session': VALID_FILE
        })

        self.train_data.start()
        self.valid_data.start()
        self.vocab_size = len(self.vocab_lookup_dict)
        # class object
        # todo: put variables as needed and place holders
        self.HERED = HERED(vocab_size=self.vocab_size, embedding_dim=EMBEDDING_DIM, query_dim=QUERY_DIM,
                           session_dim=SESSION_DIM, decoder_dim=QUERY_DIM, output_dim=EMBEDDING_DIM,
                           eoq_symbol=EOQ_SYMBOL, eos_symbol=EOS_SYMBOL, unk_symbol=UNK_SYMBOL,
                           learning_rate=self.config.learning_rate)

        self.sequence_max_length = tf.placeholder(tf.int64)
        # TODO: attention needs config.max_lenght to be not None
        self.X = tf.placeholder(tf.int64, shape=(config.batch_size, config.max_length)) #(BS,seq_len)
        self.Y = tf.placeholder(tf.int64, shape=(config.batch_size, config.max_length))

        # todo check this
        self.logits = self.HERED.inference(self.X,self.Y, self.sequence_max_length, attention=True)
        self.loss = self.HERED.get_loss(self.logits, self.Y)
        # self.loss_val = tf.placeholder(tf.float32)

        # self.softmax = self.HERED.softmax(self.logits)
        # self.accuracy = self.HERED.accuracy(self.logits, self.Y)
        #self.get_predictions = self.HERED.get_predictions(self.X)

        # Define global step for the optimizer  --- OPTIMIZER
        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.optimizer = self.get_optimizer(self.loss, self.config.learning_rate, global_step)

        some_variables = 0

        # ...
        #

    def train_model(self, batch_size=None):
        # summaries = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            log_path = self.config.summary_path + time.strftime("%Y%m%d-%H%M")
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_path)
            writer.add_graph(sess.graph)

            total_loss = 0.0
            # initialisation
            # x_batch, y_batch, seq_len = self.get_batch(dataset='train')
            # feed_dict = {
            #     self.X: x_batch,
            #     self.Y: y_batch,
            #     self.sequence_max_length : seq_len
            #
            # }
            # todo ?
            #sess.run([self.HERED.initialise], feed_dict=feed_dict)

            for iteration in range(self.config.max_steps):

                #todo:
                t1 = time.time()

                # x_batch, y_batch, seq_len = self.get_batch(dataset='train')
                # print(x_batch)
                x_batch, y_batch, seq_len = self.get_random_batch()

                print(x_batch.shape,y_batch.shape)
                feed_dict = {
                    self.X: x_batch,
                    self.Y: y_batch,
                    self.sequence_max_length : seq_len
                }
                # logits_ = sess.run([self.logits],feed_dict=feed_dict)
                # loss_value,_ = sess.run([self.loss,self.optimizer],)
                _,loss_val = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)

                t2 = time.time()

                examples_per_second = self.config.batch_size/float(t2-t1)

                # Output the training progress
                if iteration % self.config.print_every == 0:
                    print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.2f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), iteration+1,
                        int(self.config.max_steps), self.config.batch_size, examples_per_second,
                        loss_val
                    ))


                # Update the events file.
                #summary_str = sess.run(summary, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, train_step)
                #summary_writer.flush()

                #if iteration % config.checkpoint_every == 0:
                #    saver.save(sess, save_path=config.checkpoint_path)
        return sess

    def predict_model(self, sess=None):
        raise NotImplementedError
        if not sess:
            saver.restore(sess, config.checkpoint_path)
        x_batch, y_batch, seq_len = self.get_batch(dataset='valid')
        feed_dict = {
            self.X: x_batch,
            self.Y: y_batch
        }
        # self.predictions: tensor function to compute predictions given x_batch
        query_output = sess.run([self.get_predictions], feed_dict=feed_dict)
        return


    def get_batch(self, dataset):
        if dataset == 'train':
            data = self.train_data.next()
        elif dataset == 'valid':
            data = self.valid_data.next()
        else:
            raise BaseException('get_batch(): Dataset must be either "train" or "valid"')
        seq_len = data['max_length']
        prepend = np.ones((1, data['x'].shape[1]))
        x_data_full = np.concatenate((prepend, data['x']))
        x_batch = x_data_full[:seq_len]# [seq_len, embedding_dimension]
        y_batch = x_data_full[1:seq_len + 1]# [seq_len, embedding_dimension]

        print(seq_len)
        return x_batch, y_batch, seq_len


    def get_optimizer(self, loss, learning_rate, global_step, max_norm_gradient=10.0):
        """
        Optimizer with clipped gradients.

        :param loss: tensor, loss to minimize
        :param learning_rate: float, learning rate
        :param max_norm_gradient: float, max value for the gradients. Default is 10.0
        :return: the optimizer object
        """

        # Define the optimizer with default parameters set by tensorflow (the ones from the paper)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Clip gradients
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, variables = zip(*grads_and_vars)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=max_norm_gradient)
        opt = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

        # Return minimizer
        return opt

    def get_random_batch(self):
        a = np.random.randint(5000, size=(51, 7))
        b = a[1:]
        a = a[:-1]

        return a, b, 7



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH, help='Max length.')
    parser.add_argument('--buckets', type=int, default=N_BUCKETS, help='Number of buckets.')

    # Training params
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS, help='Number of steps to run trainer.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--summary_path', type=str, default='./summaries/',help='Output path for summaries.')
    parser.add_argument('--checkpoint_every', type=int, default=10,help='How often to save checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.ckpt',help='Output path for checkpoints.')
    FLAGS, unparsed = parser.parse_known_args()

    with tf.Graph().as_default():
        trainer = Train(config=FLAGS)
        trainer.train_model(batch_size=FLAGS.batch_size)
