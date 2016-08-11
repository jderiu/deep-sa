
fname = 'smiley_tweets_embedding_multilingual300M'
out_fname = 'smiley_tweets_embedding_multilingual300M_clean'

ofile = open(out_fname,'w')
for line in open(fname,'r'):
    if len(line.split('\t')) == 53:
        ofile.write(line)