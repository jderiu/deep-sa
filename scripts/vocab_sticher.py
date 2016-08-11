import cPickle
import os
from alphabet import Alphabet
import operator

data_dir= 'preprocessed_data'

fnames = [
    'vocab_en300M',
    'vocab_german40M',
    'vocab_italian_44M',
    'vocab_netherlands40M'
]

new_alphabet = Alphabet(start_feature_id=0)
new_alphabet.add('UNKNOWN_WORD_IDX')
dummy_word_idx = new_alphabet.fid

for fname in fnames:
    appfname= '{}.pickle'.format(fname)
    fname_vocab = os.path.join(data_dir,appfname)

    alphabet = cPickle.load(open(fname_vocab))
    print "alphabet", len(alphabet)
    word_freq = map(lambda x: (x[0],x[1][1]),alphabet.items())

    sorted_x = sorted(word_freq, key=operator.itemgetter(1),reverse=True)[:650000]
    print len(sorted_x)
    print sorted_x[0]

    for word,freq in sorted_x:
        new_alphabet.add(word)

print len(new_alphabet)
cPickle.dump(new_alphabet, open(os.path.join(data_dir, 'vocab_mixed_2Mwords.pickle'), 'wb'))