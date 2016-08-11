de_eng_40M = open('de_eng_n_german40M','r').readlines()
de_eng_160M = open('de_eng_n_cleaned_160M','r').readlines()

de_eng_40M = map(lambda x : x.replace('\n','').replace('\r','').split('\t'), de_eng_40M)
de_eng_160M = map(lambda x : x.replace('\n','').replace('\r','').split('\t')[-1], de_eng_160M)

ofile = open('de_eng_output.tsv','w')

output = []
for r,l in zip(de_eng_40M,de_eng_160M):
    r.append(l)
    output.append('\t'.join(r) + '\n')

ofile.writelines(output)
print 'hi'
