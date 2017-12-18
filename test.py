def _itersplit(l, splitters):
    current = []
    for item in l:
        if item in splitters:
            yield current
            current = []
        else:
            current.append(item)
    yield current

def magicsplit(l, *splitters):
    return [subl for subl in _itersplit(l, splitters) if subl]

session_file = 'data/input_model/train.ses.pkl'
import _pickle as cPickle
import numpy
data = cPickle.load(open(session_file, 'rb'))
# data = data[0:1]
session = []
allq  = []
len_data = len(data)
# max_len = 0
for i, sub_data in enumerate(data):
    print(i/len_data)
    a = magicsplit(sub_data,1,2)
    # curlen =len(i)
    # if curlen>max_len:
    #     max_len=curlen

    for query in a:
        if len(query)>50:
            query = query[0:49]
        for n, i in enumerate(query):
            if i == 0:
                query[n] = 5003
                #print(query)
        session.append(query)
    allq.append(session)
    session = []
cPickle.dump(allq,open( "allq_train.p", "wb" ))


session_file = 'data/input_model/valid.ses.pkl'
import _pickle as cPickle
import numpy
data = cPickle.load(open(session_file, 'rb'))
# data = data[0:1]
session = []
allq = []
# max_len = 0
for i, sub_data in enumerate(data):
    print(i/len(data))
    a = magicsplit(sub_data,1,2)
    # curlen =len(i)
    # if curlen>max_len:
    #     max_len=curlen

    for query in a:
        if len(query)>50:
            query = query[0:49]
        for n, i in enumerate(query):
            if i == 0:
                query[n] = 5003
        session.append(query)
    allq.append(session)
    session = []
cPickle.dump(allq,open( "allq_valid.p", "wb" ))
#
#

#
#
#
#
# '''
# for i in range(len(allq)):
#     # a = magicsplit(i,1,2)
#     curlen =len(allq[i])
#     if curlen>max_len:
#         max_len=curlen
#         print('curr_max_len',max_len)
#         print('index',i)
#
#
#
# # print(allq)0
# print(max_len)
#
# '''
# from get_batch import get_batch
# # randomlist = range(0,1000,50)
# randomlist=list(range(0,1000,50))
#
# x_batch, y_batch, max_len, randomlist = get_batch(randomlist, type='train', element=50, batch_size=64, max_len=25)
#
# print(x_batch)
# print(y_batch)
# print(max_len)
# print(randomlist)
# print('hello')
# VOCAB_FILE = '../../data/input_model/train.dict.pkl'
# vocab = cPickle.load(open(VOCAB_FILE, 'rb'))
# vocab_lookup_dict = {k: v for v, k, count in vocab}
# vocab_lookup_dict[50003] = vocab_lookup_dict[0]
# vocab_lookup_dict[0] = '<pad>'
#
