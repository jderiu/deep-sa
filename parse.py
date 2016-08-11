import cPickle
import os
import sys
import getopt

from alphabet import Alphabet
import numpy as np
from nltk.tokenize import TweetTokenizer
import parse_utils as p_utils

UNKNOWN_WORD_IDX = 0
DUMMY_WORD_IDX = 1

def convert_sentiment(sentiment):
    return {
        "positive": 2,
        "negative": 0,
        "objective-OR-neutral" : 1,
        "objective" :1,
        "UNK": np.random.randint(3)
    }.get(sentiment,1)


def load_data(fname,alphabet,ncols=4):
    tid,tweets,sentiments = [],[],[]
    tknzr = TweetTokenizer(reduce_len=True)
    n_not_available = 0
    with open(fname) as f:
        for line in f:
            splits = line.split('\t')
            tweet = splits[ncols - 1]
            sentiment = convert_sentiment(splits[ncols - 2])
            if tweet != "Not Available\n":
                tid.append(splits[0])
                tweet = p_utils.preprocess_tweet(splits[ncols - 1])
                tweet_tok = tknzr.tokenize(tweet.decode('utf-8'))
                tweets.append(tweet_tok)
                sentiments.append(int(sentiment))
            else:
                n_not_available += 1

    print "Number of not availalbe tweets:", n_not_available
    return tid,tweets,sentiments


def main(argv):
    vocab_dir = 'preprocessed_data'
    load_vocab = False
    parse_200M = True

    smiley_tweets_fname = ''
    smiley_tweets = ''
    fname_vocab = ''
    n_max_tweets = np.inf
    outdir  =''
    parse_random_tweets = False

    try:
        opts, args = getopt.getopt(argv, "v:t:m:nr", ["vocab=", "tweets=","max_tweets=","no_big="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-v", "--vocab"):
            load_vocab = True
            fname_vocab = os.path.join(vocab_dir, '{}.pickle'.format(arg))
        elif opt in ("-t", "--tweets"):
            smiley_tweets_fname = arg
            smiley_tweets = 'semeval/{}.gz'.format(arg)
            outdir = 'preprocessed_data_{}'.format(arg)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            model_dir = 'misc/{}'.format(arg)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        elif opt in ("-m", "--max_tweets"):
            n_max_tweets = int(arg)
        elif opt == '-n':
            parse_200M = False
        elif opt == '-r':
            parse_random_tweets = True



    dev2013 = "semeval/dev2013-task-B.tsv"
    dev2016 = "semeval/dev2016-task-A.tsv"
    devtest2016 = "semeval/devtest2016-task-A.tsv"
    test2013_sms = "semeval/test2013sms-task-B.tsv"
    test2013_twitter = "semeval/test2013-task-B.tsv"
    test2014_livejournal = "semeval/test2014lj-task-B.tsv"
    test2014_sarcasm = "semeval/test2014sarcasm-task-B.tsv"
    test2014_twitter = "semeval/test2014-task-B.tsv"
    test2015 = "semeval/test2015-task-B.tsv"
    test2016 = "semeval/test2016-task-A.tsv"
    train2013 = "semeval/train2013-task-B.tsv"
    train16 = "semeval/train2016-task-A.tsv"
    de_train = "semeval/de_train.tsv"
    de_test = "semeval/de_test.tsv"
    it_test = "semeval/it_test.tsv"
    it_train = "semeval/it_train.tsv"
    nl_train = "semeval/nl_train.tsv"
    nl_test = "semeval/nl_test.tsv"
    de_en_test = "semeval/de_eng_n.tsv"
    de_no_en_test = "semeval/de_no_eng_n.tsv"



    if load_vocab:
        alphabet = cPickle.load(open(fname_vocab))
        dummy_word_idx = alphabet.get('DUMMY_WORD_IDX',DUMMY_WORD_IDX)
        print "alphabet", len(alphabet)
        print 'dummy_word:',dummy_word_idx
    else:
        alphabet = Alphabet(start_feature_id=0)
        alphabet.add('UNKNOWN_WORD_IDX')
        alphabet.add('DUMMY_WORD_IDX')
        dummy_word_idx = DUMMY_WORD_IDX

    print "Loading Semeval Data"
    #ncol is the number of columns iside the files in semeval
    files = [(train2013,4),
             (dev2013,4),
             (test2013_sms,4),
             (test2013_twitter,4),
             (test2014_twitter,4),
             (test2014_livejournal,4),
             (test2014_sarcasm,4),
             (test2015,4),
             (train16,3),
             (dev2016,3),
             (devtest2016,3),
             (test2016,3),
             (de_test,4),
             (de_train,4),
             (it_test,4),
             (it_train,4),
             (nl_test,4),
             (nl_train,4),
             (de_en_test,4),
             (de_no_en_test,4),
             ]
    if parse_random_tweets:
        outdir = outdir + '_random'
        files = map(lambda x: (os.path.join('random_tweets',x),3),os.listdir('random_tweets'))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    for fname,ncols in files:
        tid, tweets, sentiments = load_data(fname,alphabet,ncols=ncols)
        print "Number of tweets:",len(tweets)

        tweet_idx = p_utils.convert2indices(tweets, alphabet, dummy_word_idx)

        basename, _ = os.path.splitext(os.path.basename(fname))
        np.save(os.path.join(outdir, '{}.tids.npy'.format(basename)), tid)
        np.save(os.path.join(outdir, '{}.tweets.npy'.format(basename)), tweet_idx)
        np.save(os.path.join(outdir, '{}.sentiments.npy'.format(basename)), sentiments)

    if parse_200M:
        print "Loading Smiley Data"
        basename, _ = os.path.splitext(os.path.basename('smiley_tweets'))
        nTweets = p_utils.store_file(
            smiley_tweets,
            os.path.join(outdir, '{}.tweets.npy'.format(basename)),
            alphabet,
            dummy_word_idx,
            sentiment_fname=os.path.join(outdir, '{}.sentiments.npy'.format(basename)),
            max_tweets=n_max_tweets
        )
        print "Number of tweets:", nTweets
        nTf = open('misc/{}/nTweets.txt'.format(smiley_tweets_fname),'wb')
        nTf.write(str(nTweets))
        nTf.close()

    cPickle.dump(alphabet, open(os.path.join(outdir, 'last_vocab.pickle'), 'wb'))

if __name__ == '__main__':
    main(sys.argv[1:])