import numpy as np
import os
import threading
from keras.utils.np_utils import to_categorical

dev2013 = "dev2013-task-B"
dev2016 = "dev2016-task-A"
devtest2016 = "devtest2016-task-A"
test2013_sms = "test2013sms-task-B"
test2013_twitter = "test2013-task-B"
test2014_livejournal = "test2014lj-task-B"
test2014_sarcasm = "test2014sarcasm-task-B"
test2014_twitter = "test2014-task-B"
test2015 = "test2015-task-B"
test2016 = "test2016-task-A"
train2013 = "train2013-task-B"
train2016 = "train2016-task-A"
de_train = "de_train"
de_test = "de_test"
it_test = "it_test"
it_train = "it_train"
nl_train = "nl_train"
nl_test = "nl_test"
de_en_test = "de_eng_n"
de_en_train = "de_no_eng_n"

def load_supervised(data_dir,lang='multi',max_len=140):
    np.random.seed(12345)
    if lang == 'multi':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016,de_train,it_train,nl_train]
        validation_files = [test2015,it_test,de_test]
        test_files = [test2015,test2016,nl_test,it_test,de_test]
    elif lang == 'en':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016,]
        validation_files = [test2016]
        test_files = [test2015,test2016]
    elif lang == 'de':
        training_files = [de_train]
        validation_files = [de_test]
        test_files = [de_test]
    elif lang == 'it':
        training_files = [it_train]
        validation_files = [it_test]
        test_files = [it_test]
    elif lang == 'nl':
        training_files = [nl_train]
        validation_files = [nl_test]
        test_files = [nl_test]
    elif lang == 'en_valid':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016, de_train, it_train, nl_train]
        validation_files = [test2016]
        test_files = [test2015, test2016]
    elif lang == 'de_valid':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016, de_train, it_train, nl_train]
        validation_files = [de_test]
        test_files = [de_test]
    elif lang == 'it_valid':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016, de_train, it_train, nl_train]
        validation_files = [it_test]
        test_files = [it_test]
    elif lang == 'nl_valid':
        training_files = [dev2013, train2013, dev2016, devtest2016, train2016, de_train, it_train, nl_train]
        validation_files = [nl_test]
        test_files = [nl_test]
    elif lang == 'de_eng':
        training_files = [de_en_train]
        validation_files = [de_en_test]
        test_files = [de_en_test]


    Tids_train = None
    X_train = None
    y_train = None

    for fname in training_files:
        tids = np.load(os.path.join(data_dir, '{}.tids.npy').format(fname))
        tweets = np.load(os.path.join(data_dir, '{}.tweets.npy').format(fname))
        sentiments = np.load(os.path.join(data_dir, '{}.sentiments.npy').format(fname))

        if Tids_train is None:
            Tids_train = tids
        else:
            Tids_train = np.concatenate((Tids_train, tids), axis=0)

        if X_train is None:
            X_train = tweets
        else:
            X_train = np.concatenate((X_train, tweets), axis=0)

        if y_train is None:
            y_train = sentiments
        else:
            y_train = np.concatenate((y_train, sentiments), axis=0)

    shuffled_train = np.random.permutation(np.concatenate((X_train,y_train[:,None]),axis=1))
    X_train,y_train = shuffled_train[:,0:-1],shuffled_train[:,-1]

    Tids_valid = None #np.load(os.path.join(data_dir, '{}.tids.npy').format(validation_file))
    X_valid = None #None#np.load(os.path.join(data_dir, '{}.tweets.npy').format(validation_file))
    y_valid = None #np.load(os.path.join(data_dir, '{}.sentiments.npy').format(validation_file))

    for fname in validation_files:
        tids = np.load(os.path.join(data_dir, '{}.tids.npy').format(fname))
        tweets = np.load(os.path.join(data_dir, '{}.tweets.npy').format(fname))
        sentiments = np.load(os.path.join(data_dir, '{}.sentiments.npy').format(fname))

        if Tids_valid is None:
            Tids_valid = tids
        else:
            Tids_valid = np.concatenate((Tids_valid, tids), axis=0)

        if X_valid is None:
            X_valid = tweets
        else:
            X_valid = np.concatenate((X_valid, tweets), axis=0)

        if y_valid is None:
            y_valid = sentiments
        else:
            y_valid = np.concatenate((y_valid, sentiments), axis=0)

    test_sets = []
    for fname in test_files:
        tids = np.load(os.path.join(data_dir, '{}.tids.npy').format(fname))
        tweets = np.load(os.path.join(data_dir, '{}.tweets.npy').format(fname))
        sentiments = np.load(os.path.join(data_dir, '{}.sentiments.npy').format(fname))
        test_sets.append((tids,tweets, sentiments, fname))

    return (Tids_train,X_train, y_train), (Tids_valid,X_valid, y_valid), test_sets


class ThreadSafeIterator:
    """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadSafeIterator(f(*a, **kw))
    return g


class DistantDataIterator(object):
    def __init__(self, data_dir,fname_tweet,fname_sentiment, max_tweets=np.inf,max_len = 140):
        self.ftweets = fname_tweet
        self.fsentiment = fname_sentiment
        self.data_dir = data_dir
        self.max_tweets = max_tweets
        self.batch_lock = threading.Lock()
        self.chunk_lock = threading.Lock()
        self.max_len = max_len

    def flow(self, batch_size=100):

        next_chunk = True

        while True:
            file_tweets = open(os.path.join(self.data_dir, self.ftweets), 'rb')
            file_sentiments = open(os.path.join(self.data_dir, self.fsentiment), 'rb')
            counter = 0
            while True:
                if counter > self.max_tweets:
                    break
                try:
                    tweets = np.load(file_tweets)
                    sentiments = np.load(file_sentiments)

                    padding = self.max_len - tweets.shape[1]
                    if padding > 0:
                        pad_matrix = np.ones((tweets.shape[0],padding),dtype='int')
                        tweets = np.concatenate((tweets,pad_matrix),axis=1)

                except:
                    break
                counter += tweets.shape[0]
                chunk_size = tweets.shape[0]
                for i in xrange(0,chunk_size,batch_size):
                        yield (tweets[i:min(i+batch_size,chunk_size)],to_categorical(sentiments[i:min(i+batch_size,chunk_size)],2))

            file_tweets.close()
            file_sentiments.close()


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


