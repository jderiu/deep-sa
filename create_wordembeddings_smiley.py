import gzip
import logging
import sys
import os

from gensim import models
from nltk import TweetTokenizer
from collections import defaultdict
from parse_utils import preprocess_tweet


MAX_TW_LANG = 50000000
max_for_lang = {
    'en' : 100000000,
    'es' : 60000000,
    'it' : 33950631,
    'de' : 33035519,
    'fr' : 41348331,
    'nl' : 28643776
}



class MySentences(object):
    def __init__(self, files):
        self.files = files
        self.tknzr = TweetTokenizer()

    def max_reached(self,language_tags):
        all_max = True
        for lang in max_for_lang.keys():
            for sent in ['positive','negative']:
                tag = '{}_{}'.format(lang, sent)
                curr_is_max = language_tags[tag] >= max_for_lang[lang]
                all_max &= curr_is_max
        return all_max


    def __iter__(self):
        language_tags = defaultdict(lambda: 0)
        for (fname) in self.files:
             for line in open(fname,'rb'):
                if self.max_reached(language_tags):
                    return

                splits = line.split('\t')
                lang_tag = splits[0].strip()
                sent_tag = splits[4].strip()
                tag = '{}_{}'.format(lang_tag,sent_tag)
                if language_tags[tag] < max_for_lang[lang_tag]:
                     language_tags[tag] += 1
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
    directory = 'smiley_tweets_full'
    listing = map(lambda x: os.path.join(directory,x),os.listdir(directory))


    sentences = MySentences(files=listing)
    model = models.Word2Vec(sentences, size=52, window=5, min_count=10, workers=16,sg=1,sample=1e-5,hs=1)
    model.save_word2vec_format('embeddings/smiley_tweets_embedding_multilingual{}'.format(input_fname),binary=False)

if __name__ == '__main__':
    main()