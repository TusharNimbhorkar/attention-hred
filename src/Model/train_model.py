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
from get_batch import get_batch
import random
import logging
logging.basicConfig(filename='output.log',level=logging.DEBUG)


# todo: put this stuff in arg.parse as well
LEARNING_RATE = 1e-4
HIDDEN_LAYERS = 1
BATCH_SIZE = 60
MAX_LENGTH = 10
N_BUCKETS = 20
MAX_STEPS = 10000000
VOCAB_SIZE = 50003
random_seed = 1234
UNK_SYMBOL = 5003
EOQ_SYMBOL = 1
EOS_SYMBOL = 2
EMBEDDING_DIM = 300
QUERY_DIM = 1000
SESSION_DIM = 1500
VOCAB_FILE = '../../data/input_model/train.dict.pkl'
TRAIN_FILE = '../../data/input_model/train.ses.pkl'
VALID_FILE = '../../data/input_model/valid.ses.pkl'
train_file = '../../data/new_new_batch/allq_train.p'
valid_file = '../../data/new_new_batch/allq_train.p'



class Train(object):
    def __init__(self, config=None):

        self.config = config
        self.vocab = cPickle.load(open(VOCAB_FILE, 'rb'))
        self.vocab_lookup_dict = {k: v for v, k, count in self.vocab}
        self.vocab_lookup_dict[50003] = self.vocab_lookup_dict[0]
        self.vocab_lookup_dict[0] = '<pad>'

        # self.train_data, self.valid_data = data_iterator.get_batch_iterator(np.random.RandomState(random_seed), {
        #     'eoq_sym': EOQ_SYMBOL,
        #     'eos_sym': EOS_SYMBOL,
        #     'sort_k_batches': config.buckets,
        #     'bs': config.batch_size,
        #     'train_session': TRAIN_FILE,
        #     'seqlen': config.max_length,
        #     'valid_session': VALID_FILE
        # })
        # todo remove [0:200] for full training set
        self.train_data = cPickle.load(open(train_file, 'rb'))
        self.valid_data = cPickle.load(open(valid_file, 'rb'))
        # print('getBatch', len(data))

        # self.train_data.start()
        # self.valid_data.start()
        self.vocab_size = len(self.vocab_lookup_dict)
        # todo: put variables as needed and place holders
        #self.sequence_max_length = tf.placeholder(tf.int64)
        # TODO: attention needs config.max_lenght to be not None <---------- check this !!!
        self.X = tf.placeholder(tf.int64, shape=(None, config.max_length)) #(BS,seq_len)
        self.Y = tf.placeholder(tf.int64, shape=(None, config.max_length))

        # class object
        self.HERED = HERED(vocab_size=self.vocab_size, embedding_dim=config.embedding_dim, query_dim=config.query_dim,
                           session_dim=config.session_dim, decoder_dim=config.query_dim,
                           output_dim=config.output_dim,
                           eoq_symbol=config.eoq_symbol, eos_symbol=config.eos_symbol, unk_symbol=config.unk_symbol,
                           learning_rate=self.config.learning_rate, hidden_layer=config.hidden_layer,
                           batch_size=tf.shape(self.X)[0])


        self.logits = self.HERED.inference(self.X,self.Y, self.X.shape[1], attention=False)  # <--- set attention here
        self.loss = self.HERED.get_loss(self.logits, self.Y)
        # self.loss_val = tf.placeholder(tf.float32)

        # self.softmax = self.HERED.softmax(self.logits)
        # self.accuracy = self.HERED.accuracy(self.logits, self.Y)
        #self.get_predictions = self.HERED.get_predictions(self.X)

        # Define global step for the optimizer  --- OPTIMIZER
        global_step = tf.Variable(0, name = 'global_step', trainable=False, dtype=tf.int32)
        tf.add_to_collection('global_step', global_step)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)


    def get_length(self, sequence):
        length = np.sum(np.sign(np.abs(sequence)),1)
        return length

    def get_accuracy(self, predictions, Y):
        length = self.get_length(Y)
        predictions = predictions[0]
        correct = 0
        for i in range (len(predictions)):
            correct += np.sum(np.equal( Y[i][:length[i]], predictions[i][:length[i]]).astype(float))
        return correct/float(np.sum(length))

    def train_model(self, batch_size=None, restore = False):

        # batch parameters,train
        #train_list = list(range(0, len(self.train_data)-150, batch_size))
        train_list = list(range(0, len(self.train_data)))


        # summaries = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        if restore ==True:
            train_list = cPickle.load(open("train_list.p", 'rb'))
            saver = tf.train.Saver()

        with tf.Session() as sess:
            if restore == False:

                sess.run(tf.global_variables_initializer())
                global_step = 0

                total_loss = 0.0

                self.config.max_steps = int((len(self.train_data)-150)/self.config.batch_size)
            else:
                print(self.config.checkpoint_path)
                saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
                global_step = tf.get_collection_ref('global_step')[0]
                global_step= sess.run(global_step)

            log_path = self.config.summary_path
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_path)
            writer.add_graph(sess.graph)
            # TODO check the train list for None

            for iteration in range(global_step,  self.config.max_steps):

                #todo:
                t1 = time.time()

                # x_batch, y_batch, seq_len = self.get_batch(dataset='train')
                # print(x_batch)
                # x_batch, y_batch, seq_len = self.get_random_batch()
                random_element = random.choice(train_list)

                x_batch, y_batch, seq_len, train_list = get_batch(train_list,self.train_data, type='train', element=random_element,
                                                                  batch_size=self.config.batch_size,
                                                                  max_len=self.config.max_length, eoq=self.HERED.eoq_symbol)
                # if iteration == 2:
                #     break
                feed_dict = {
                    self.X: x_batch,
                    self.Y: y_batch
                }
                # logits_ = sess.run([self.logits],feed_dict=feed_dict)
                # loss_value,_ = sess.run([self.loss,self.optimizer],)
                _, loss_val, summ= sess.run([self.optimizer, self.loss, summaries], feed_dict=feed_dict)
                writer.add_summary(summ, iteration)

                t2 = time.time()

                examples_per_second = self.config.batch_size/float(t2-t1)

                # Output the training progress
                if iteration % self.config.print_every == 0:
                    print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.2f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), iteration+1,
                        int(self.config.max_steps), self.config.batch_size, examples_per_second,
                        loss_val
                    ))
                    logging.debug("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.2f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), iteration+1,
                        int(self.config.max_steps), self.config.batch_size, examples_per_second,
                        loss_val
                    ))

                if iteration %  self.config.print_every  == 0: #self.config.validate_every
                    valid_list = list(range(0, len(self.valid_data) - 150, batch_size))
                    random_element = random.choice(valid_list)
                    x_batch, y_batch, _, _ = get_batch(valid_list, self.valid_data, type='train',
                                                                      element=random_element,
                                                                      batch_size=self.config.batch_size,
                                                                      max_len=self.config.max_length, eoq=self.HERED.eoq_symbol)

                    predictions = sess.run([self.HERED.validation(X = self.X, Y= self.Y, attention=False)], feed_dict={self.X: x_batch, self.Y: y_batch})
                    accuracy = self.get_accuracy(predictions, y_batch)

                    # print(self.get_length(y_batch))
                    # # print(np.sum(mask,1))
                    # print(mask)
                    # print(np.sum(mask,1))
                    print('accuracy ' + str(accuracy))
                    logging.debug('accuracy ' + str(accuracy))


                    # summary = sess.run(summaries,
                    #                    feed_dict={ self.loss: loss_val})
                    # writer.add_summary(summary, global_step=iteration)


                # Update the events file.
                #summary_str = sess.run(summary, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, train_step)
                #summary_writer.flush()

                #if iteration+1 % self.config.checkpoint_every == 0:
                #    saver.save(sess, save_path= self.config.checkpoint_path ,global_step=iteration)
                #    cPickle.dump(train_list, open("train_list.p", "wb"))
        return sess

    def restore_training(self):
        train_file = cPickle.load(open("train_list.p", 'rb'))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.config.checkpoint_path))

    def predict_model(self, sess=None):

        if not sess:
            #RESTORE TRAIN LIST
            # batch parameters,train
            test_list = list(range(0, len(self.train_data) - 50, self.config.batch_size))[0:100]
            random_element = random.choice(test_list)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, self.config.checkpoint_path)
                x_batch, y_batch, seq_len = get_batch(test_list,self.train_data, type='test', element=random_element,
                                                                  batch_size=self.config.batch_size,
                                                                  max_len=50)
                feed_dict = {
                    self.X: x_batch,
                    self.Y: y_batch
                }
                # self.predictions: tensor function to compute predictions given x_batch
                query_output = sess.run([self.HERED.get_predictions], feed_dict=feed_dict)
        return


    def get_batch_old(self, dataset):
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
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help='Embedding dimensions.')
    parser.add_argument('--query_dim', type=int, default=QUERY_DIM, help='Query encoder dims')
    parser.add_argument('--session_dim', type=int, default=SESSION_DIM, help='Session encoder dims.')
    parser.add_argument('--decoder_dim', type=int, default=QUERY_DIM, help='Decoder dims.')
    parser.add_argument('--output_dim', type=int, default=EMBEDDING_DIM, help='Output embedding dims.')
    parser.add_argument('--eoq_symbol', type=int, default=EOQ_SYMBOL, help='End of query symbol.')
    parser.add_argument('--eos_symbol', type=int, default=EOS_SYMBOL, help='End of session symbol.')
    parser.add_argument('--unk_symbol', type=int, default=UNK_SYMBOL, help='Unknown symbol.')
    parser.add_argument('--hidden_layer', type=int, default=HIDDEN_LAYERS, help='Number of hidden layers.')


    # Training params
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS, help='Number of steps to run trainer.')

    # Misc params
    parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    parser.add_argument('--summary_path', type=str, default='./summaries/',help='Output path for summaries.')
    parser.add_argument('--checkpoint_every', type=int, default=1000,help='How often to save checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/model.ckpt',help='Output path for checkpoints.')
    FLAGS, unparsed = parser.parse_known_args()

    with tf.Graph().as_default():
        trainer = Train(config=FLAGS)
        trainer.train_model(batch_size=FLAGS.batch_size)
        # trainer.predict_model()
