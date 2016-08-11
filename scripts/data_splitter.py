import gzip
import getopt
import sys
import os

from nltk import TweetTokenizer
from collections import defaultdict
from tqdm import tqdm

#python data_splitter.py -e 50000000 -s 20000000 -i 330000000 -d 33000000 -f 20000000 -n 30000000 -o filtered_balanced_300M_en100M
MAX_TW_LANG = 50000000

max_for_lang = {
    'en' : 0,
    'es' : 0,
    'it' : 0,
    'de' : 0,
    'fr' : 0,
    'nl' : 0
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
             for line in gzip.open(fname,'rb'):
                if self.max_reached(language_tags):
                    return

                splits = line.split('\t')
                lang_tag = splits[0].strip()
                sent_tag = splits[4].strip()
                tag = '{}_{}'.format(lang_tag,sent_tag)
                if language_tags[tag] < max_for_lang[lang_tag]:
                     language_tags[tag] += 1
                     yield line


def main(argv):
    out_file = ''
    in_file = ''
    try:
        opts, args = getopt.getopt(argv, "e:s:i:d:n:f:o:q:", ["en=","es=","it=","de=","nl=","fr=","ofile=","ifile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-e", "--en"):
            max_for_lang['en'] = int(arg)
        elif opt in ("-s", "--es"):
            max_for_lang['es'] = int(arg)
        elif opt in ("-i", "--it"):
            max_for_lang['it'] = int(arg)
        elif opt in ("-d", "--de"):
            max_for_lang['de'] = int(arg)
        elif opt in ("-n", "--nl"):
            max_for_lang['nl'] = int(arg)
        elif opt in ("-f", "--fr"):
            max_for_lang['fr'] = int(arg)
        elif opt in ("-o", "--ofile"):
            out_file = '{}.gz'.format(arg)
        elif opt in ("-q", "--ifile"):
            in_file = 'semeval/{}.gz'.format(arg)

    #unsupervised data
    directory = 'smiley_tweets_full'
    #listing = map(lambda x: os.path.join(directory,x),os.listdir(directory))
    files = [in_file]

    out_path = os.path.join('semeval',out_file)
    out_file = gzip.open(out_path,'w')

    sentences = MySentences(files=files)
    for sent in tqdm(sentences):
        out_file.write(sent)

if __name__ == '__main__':
    main(sys.argv[1:])

