import cPickle
import os
import numpy as np
from alphabet import Alphabet
import getopt
import sys

from utils import load_glove_vec


def main(argv):
    np.random.seed(123)
    data_dir = 'preprocessed_data'
    emb_path = 'embeddings/smiley_tweets_embedding_multilingual300M'
    emb_name = 'smiley_tweets_embedding_mixed2M_words'
    fname_vocab = os.path.join(data_dir, 'vocab_reduced.pickle')
    multi_emb_path = [
        'embeddings/smiley_tweets_embedding_netherlands_300M',
        'embeddings/smiley_tweets_embedding_german_300M',
        'embeddings/smiley_tweets_embedding_italian_300M',
        'embeddings/smiley_tweets_embedding_english_590M',
    ]


    try:
        opts, args = getopt.getopt(argv, "v:e:", ["vocab=", "embedding="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-v", "--vocab"):
            fname_vocab = os.path.join(data_dir, '{}.pickle'.format(arg))
        elif opt in ("-e", "--embedding"):
            emb_path = 'embeddings/{}'.format(arg)
            emb_name = arg

    #get vocabulary
    print(fname_vocab)
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)

    word2vec = {}
    #get embeddings
    for p in multi_emb_path:
        fname,delimiter,ndim = (p,' ', 52)
        word2vec.update(load_glove_vec(fname, words, delimiter, ndim))

    print len(word2vec.keys())
    ndim = len(word2vec[word2vec.keys()[0]])
    print 'ndim', ndim

    random_words_count = 0
    vocab_emb = np.zeros((len(alphabet) + 1, ndim), dtype='float32')
    for word,(idx,freq) in alphabet.iteritems():
        word_vec = word2vec.get(word, None)
        if word_vec is None or word_vec.shape[0] != 52:
          word_vec = np.random.uniform(-0.25, 0.25, ndim)
          random_words_count += 1
        vocab_emb[idx] = word_vec
    print 'random_words_count', random_words_count
    print vocab_emb.shape
    outfile = os.path.join(data_dir, 'emb_{}.npy'.format(emb_name))
    print outfile
    np.save(outfile, vocab_emb)

if __name__ == '__main__':
  main(sys.argv[1:])
