from alphabet import Alphabet
import os
import cPickle
from tqdm import tqdm
import getopt
import sys

UNKNOWN_WORD_IDX = 0
DUMMY_WORD_IDX = 1
outdir = 'preprocessed_data'
in_file = ''
out_file = ''
threshold = 10

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:t:", ["ifile=",'threshold='])
except getopt.GetoptError:
    print 'test.py -i <inputfile> -o <outputfile>'
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i", "--ifile"):
        in_file = '{}.pickle'.format(arg)
        out_file = '{}_reduced.pickle'.format(arg)
    elif opt in ("-t", "--threshold"):
        threshold = int(arg)


fname_vocab = os.path.join(outdir, in_file)
alphabet = cPickle.load(open(fname_vocab))
dummy_word_idx = alphabet.get('DUMMY_WORD_IDX', DUMMY_WORD_IDX)
print "alphabet", len(alphabet)
print 'dummy_word:', dummy_word_idx

new_alphabet = Alphabet(start_feature_id=0)
new_alphabet.add('UNKNOWN_WORD_IDX')

for word, (idx,freq) in tqdm(alphabet.items()):
    if freq > threshold:
        new_alphabet.add(word)

new_alphabet.add('DUMMY_WORD_IDX"')
print "alphabet", len(new_alphabet)
cPickle.dump(new_alphabet, open(os.path.join(outdir, out_file), 'wb'))