from alphabet import Alphabet
from collections import Counter
import cPickle
import os

vocab_dir = 'preprocessed_data'
arg = 'vocab_balanced_300M'
fname_vocab = os.path.join(vocab_dir, '{}.pickle'.format(arg))

alphabet = cPickle.load(open(fname_vocab))
dummy_word_idx = alphabet.get('DUMMY_WORD_IDX' ,1)
print "alphabet", len(alphabet)
print 'dummy_word:' ,dummy_word_idx

frequencies = map(lambda x: x[1],alphabet.values())
min_freq = min(frequencies)
max_freq = max(frequencies)

print min_freq
print max_freq

c = Counter(frequencies)
print len(alphabet) - sum(map(lambda x: x[1],c.most_common(20)))


print c.most_common(20)