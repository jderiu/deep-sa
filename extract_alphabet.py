from alphabet import Alphabet
from utils import load_glove_vec
import os
import cPickle
from tqdm import tqdm

def main():
    outdir = "preprocessed_data"
    out_file = 'vocal_wembext.pickle'
    fname, delimiter, ndim = ('embeddings/smiley_tweets_embedding_multilingual300M', ' ', 52)
    word2vec = load_glove_vec(fname, {}, delimiter, ndim)

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')
    alphabet.add('DUMMY_WORD_IDX')
    dummy_word_idx = alphabet.get('DUMMY_WORD_IDX')

    for token in word2vec.keys():
        alphabet.add(token)

    print 'Alphabet before purge:', len(alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, out_file), 'wb'))


if __name__ == '__main__':
  main()
