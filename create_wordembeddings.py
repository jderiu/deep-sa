import gzip
import logging
import sys
import os

from gensim import models
from nltk import TweetTokenizer
from collections import defaultdict
from parse_utils import preprocess_tweet


MAX_TW_LANG = 200000000

class MySentences(object):
    def __init__(self, listings,gzpFiles):
        self.listings = listings
        self.gzip_files = gzpFiles
        self.tknzr = TweetTokenizer()

    def __iter__(self):
        for files in self.listings:
            file_done = False
            counter = 0
            for (fname) in files:
                if file_done:
                    break
                for line in open(fname,'rb'):
                    if counter >= MAX_TW_LANG:
                        file_done = True
                        break

                    counter += 1
                    tweet = line.split('\t')[-1]
                    tweet = preprocess_tweet(tweet)
                    tweet = self.tknzr.tokenize(tweet.decode('utf-8'))
                    yield filter(lambda word: ' ' not in word, tweet)

        counter = 0
        for (fname) in self.gzip_files:
            for line in gzip.open(fname, 'rb'):
                if counter >= MAX_TW_LANG:
                    return

                counter += 1
                tweet = line.split('\t')[-1]
                tweet = preprocess_tweet(tweet)
                tweet = self.tknzr.tokenize(tweet.decode('utf-8'))
                yield filter(lambda word: ' ' not in word, tweet)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    input_fname = ''
    if len(sys.argv) > 1:
        input_fname = sys.argv[1]

    #unsupervised data
    directories = [
        'smiley_tweets_nl_full',
        'smiley_tweets_de_full',
        'smiley_tweets_it_full',
    ]
    files = ['filtered_en_balanced580M.gz']

    listing = []
    for dir in directories:
        listing.append(map(lambda x: os.path.join(dir,x),os.listdir(dir)))

    sentences = MySentences(listings=listing,gzpFiles=files)
    model = models.Word2Vec(sentences, size=52, window=5, min_count=10, workers=16,sg=1,sample=1e-5,hs=1)
    model.save_word2vec_format('embeddings/smiley_tweets_embedding_multilingual{}'.format(input_fname),binary=False)

if __name__ == '__main__':
    main()