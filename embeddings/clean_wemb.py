from tqdm import tqdm
fname = 'smiley_tweets_embedding_multilingual300M'
out_fname = 'smiley_tweets_embedding_multilingual300M_clean'

ofile = open(out_fname,'w')
ifile = open(fname,'r')
header = ifile.readline()
ofile.write(header)
for line in tqdm(ifile):
    l = len(line.split(' '))
    if l == 53:
        ofile.write(line)