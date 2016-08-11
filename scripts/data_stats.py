from collections import Counter
import numpy as np


de_file = '../semeval/de{}.tsv'
it_file = '../semeval/it{}.tsv'
nl_file = '../semeval/nl{}.tsv'

de_en_test = "../semeval/de_no_eng.tsv"
de_en_test_ = "../semeval/de_no_eng_n.tsv"

files = [de_file,it_file,nl_file]

data = open(de_en_test,'r').readlines()
data = map(lambda x : x.replace('\n','').split('\t'),data)
data = map(lambda x: '\t'.join([x[0],x[3],x[1],x[2]]) + '\n',data)

open(de_en_test_,'w').writelines(data)








