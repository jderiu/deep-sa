import cPickle
import gzip
import os
from alphabet import Alphabet
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from parse_utils import preprocess_tweet,convert_sentiment
import numpy as np
import getopt
import sys
from utils import load_glove_vocabulary

def main(argv):
    outdir = "preprocessed_data"

    out_file = ''
    out_reduced = ''
    in_file = ''
    max_tweets = np.inf
    fwemb_vocabulary = None
    try:
        opts, args = getopt.getopt(argv, "i:o:f:m:", ["ifile=","ofile=","wfilter",'maxTweets'])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--ofile"):
            out_file = '{}.pickle'.format(arg)
            out_reduced = '{}_reduced.pickle'.format(arg)
        elif opt in ("-i", "--ifile"):
            in_file = 'semeval/{}.gz'.format(arg)
        elif opt in ('-f', '--wfilter'):
            fwemb_vocabulary = load_glove_vocabulary('embeddings/{}'.format(arg),' ')
        elif opt in ('-m','--maxTweets'):
            max_tweets = int(arg)

    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #unsupervised data
    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    dummy_word_idx = alphabet.fid

    tknzr = TweetTokenizer(reduce_len=True)
    fnames_gz = [in_file]

    counter = 0

    for fname in fnames_gz:
        with gzip.open(fname,'r') as f:
            for tweet in tqdm(f):
                tweet = tknzr.tokenize(preprocess_tweet(tweet))
                for token in tweet:
                    if fwemb_vocabulary:
                        if token in fwemb_vocabulary:
                            alphabet.add(token)
                    else:
                        alphabet.add(token)
                counter += 1
                if (counter%1000000) == 0:
                    print 'Processed tweets: {}'.format(counter)
                    print 'Alphabet Lenght: {}'.format(len(alphabet))
                if counter > max_tweets:
                    break
        print len(alphabet)

    print 'Alphabet before purge:',len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, out_file), 'wb'))

    for word, (idx, freq) in tqdm(alphabet.items()):
        if freq > 10:
            alphabet.add(word)

    alphabet.add('DUMMY_WORD_IDX"')
    print "Alphabet after purge:", len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, out_reduced), 'wb'))

if __name__ == '__main__':
    main(sys.argv[1:])