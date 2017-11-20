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

# Path to get batch iterator
sys.path.insert(0, '../sordoni/')
import data_iterator
from model import HERED

# todo: put this stuff in arg.parse as well
BATCH_SIZE = 50
MAX_LENGTH = 50
N_BUCKETS = 20
MAX_STEPS = 10000000
VOCAB_SIZE = 50003
random_seed = 1234
UNK_SYMBOL = 0
EOQ_SYMBOL = 1
EOS_SYMBOL = 2

VOCAB_FILE = '../../data/input_model/train.dict.pkl'
TRAIN_FILE = '../../data/input_model/train.ses.pkl'
VALID_FILE = '../../data/input_model/valid.ses.pkl'


class Train(object):
    def __init__(self):
        self.vocab = cPickle.load(open(VOCAB_FILE, 'rb'))
        self.vocab_lookup_dict = {k: v for v, k, count in self.vocab}

        self.train_data, self.valid_data = data_iterator.get_batch_iterator(np.random.RandomState(random_seed), {
            'eoq_sym': EOQ_SYMBOL,
            'eos_sym': EOS_SYMBOL,
            'sort_k_batches': FLAGS.buckets,
            'bs': FLAGS.buckets,
            'train_session': TRAIN_FILE,
            'seqlen': FLAGS.max_length,
            'valid_session': VALID_FILE
        })

        self.train_data.start()
        self.valid_data.start()
        self.vocab_size = len(self.vocab_lookup_dict)
        # class object
        # todo: put variables as needed and place holders
        self.HERED = HERED()
        self.X = tf.placeholder(tf.int64, shape=(None, None))
        self.Y = tf.placeholder(tf.int64, shape=(None, None))

        # init = tf.global_variables_initializer()
        # summaries = tf.summary.merge_all()
        # sess = tf.Session()
        # sess.run(init)

        # Define global step for the optimizer  --- OPTIMIZER
        #global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        #optimizer = self.get_optimizer(loss, learning_rate, global_step)

        some_variables = 0

        # ...
        #

    def train_model(self, batch_size=None):

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            total_loss = 0.0

            for iteration in range(MAX_STEPS):

                x_batch, y_batch, seq_len = self.get_batch(train_data=self.train_data)

        return

    def get_batch(self, train_data):

        data = train_data.next()
        seq_len = data['max_length']
        prepend = np.ones((1, data['x'].shape[1]))
        x_data_full = np.concatenate((prepend, data['x']))
        x_batch = x_data_full[:seq_len]
        y_batch = x_data_full[1:seq_len + 1]


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


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                        help='Max length.')
    parser.add_argument('--buckets', type=int, default=N_BUCKETS,
                        help='Number of buckets.')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS,
                        help='Number of steps to run trainer.')
    FLAGS, unparsed = parser.parse_known_args()

    with tf.Graph().as_default():
        trainer = Train()
        trainer.train_model(batch_size=FLAGS.batch_size)
