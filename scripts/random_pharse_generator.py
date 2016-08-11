import cPickle
import os

import numpy as np


def main():
    data_dir = 'preprocessed_data'

    fname_vocab = os.path.join(data_dir, 'vocab_mixed_2Mwords.pickle')
    alphabet = cPickle.load(open(fname_vocab))
    words = alphabet.keys()
    print "Vocab size", len(alphabet)

    n_twwet = 250
    tweet_len = 10
    for line in open('phrases'):
        print line
        line = line.replace('\n','')
        phrase = line.split('\t')[0]
        sentiment = line.split('\t')[1]
        print phrase
        phrase = phrase.replace('\n','').replace('\r','').split(' ')
        outfile = open('random_tweets/random_tweet_{}.tsv'.format('_'.join(phrase)),'w')
        out_lines = []
        for i in xrange(n_twwet):
            tweet = np.random.choice(words,tweet_len,True)
            #idx = np.random.choice(xrange(tweet_len - len(line) - 1),2,False)

            idx = np.random.randint(0,tweet_len - len(phrase) - 1)
            idx = xrange(idx,idx + len(phrase))
            for k,j in enumerate(idx):
                tweet[j] = phrase[k]
            out_tweet = ' '.join(tweet.tolist())
            out_line = [str(i),sentiment,out_tweet]
            out_line = '\t'.join(out_line)
            out_lines.append(out_line.encode('utf-8') + '\n')
        outfile.writelines(out_lines)
        outfile.close()

if __name__ == '__main__':
  main()
