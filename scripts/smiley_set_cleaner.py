from alphabet import Alphabet
import os
import cPickle
from tqdm import tqdm
import getopt
import sys
import gzip

outdir = 'semeval'
in_file = ''
out_file = ''
threshold = 10

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:", ["ifile=",'ofile='])
except getopt.GetoptError:
    print 'test.py -i <inputfile> -o <outputfile>'
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i", "--ifile"):
        in_file = '{}/{}.gz'.format(outdir,arg)
    elif opt in ("-o", "--ofile"):
        out_file = '{}/{}.gz'.format(outdir,arg)

ofile = gzip.open(out_file,'w')

for line in tqdm(gzip.open(in_file,'r')):
    lang_tag = line.split('\t')[0]
    if lang_tag in ['de','en','it','nl']:
        ofile.write(line)

ofile.close()