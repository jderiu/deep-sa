from alphabet import Alphabet
import os
import cPickle

UNKNOWN_WORD_IDX = 0
DUMMY_WORD_IDX = 1
outdir = 'preprocessed_data'

fname_vocab = os.path.join(outdir, 'vocab.pickle')
new_alphabet = cPickle.load(open(fname_vocab))
dummy_word_idx = new_alphabet.get('DUMMY_WORD_IDX', DUMMY_WORD_IDX)
print "alphabet", len(new_alphabet)
print 'dummy_word:', dummy_word_idx

alphabet = Alphabet(start_feature_id=0)
alphabet.add('UNKNOWN_WORD_IDX')
alphabet.add('DUMMY_WORD_IDX')
dummy_word_idx = DUMMY_WORD_IDX