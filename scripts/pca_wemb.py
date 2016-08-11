from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Courier')
matplotlib.rc('text', usetex='false')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import pylab as py
import shelve
from adjustText import adjust_text
import random

print('Load Words')
words = map(lambda x: x.replace('\n', '').lower(), open('mix_words', 'r').readlines())
emb_before = shelve.open('tmp/emb_before',protocol=2)
emb_after = shelve.open('tmp/emb_after',protocol=2)

if False:
    print('Load After Model')
    model_after = Word2Vec.load_word2vec_format('../embeddings/smiley_tweets_embedding_multilingual300M_updated', binary=False)
    print('Load Before Model')
    model_before = Word2Vec.load_word2vec_format('../embeddings/smiley_tweets_embedding_multilingual300M_clean', binary=False)

    for i,word in enumerate(words):
        try:
            xb = model_before[word]
            xa = model_after[word]
            emb_before[word] = xb
            emb_after[word] = xa

        except:
            print('Word {} does not exist in model'.format(word))



X_before = np.zeros((len(words), 52))
X_after = np.zeros((len(words), 52))
for i,word in enumerate(words):
    try:
        X_before[i] = emb_before[word]
        X_after[i] = emb_after[word]
    except:
        words.remove(word)
        print('Word {} does not exist in shelve'.format(word))


print('PCA before')
kmeans = KMeans(n_clusters=4)
c = kmeans.fit_predict(X_before)
pca_before = PCA(n_components=2,whiten=True)
X_before = pca_before.fit_transform(X_before)
texts = []
xs,ys = [],[]

print c
colors = ['red','blue','green','black']

for i in xrange(10):
    for idx in xrange(len(words)):
        word = words[idx]
        x = X_before[idx]
        xs.append(x[0])
        ys.append(x[1])
        texts.append(plt.text(x[0]+(random.random()-0.5)/2.5,x[1]+(random.random()-0.5)/2.5,word,fontweight='bold',fontsize=16,color=colors[c[idx]]))
        #py.scatter(x[0], x[1], s=50,c=colors[c[idx]])
    plt.axis((min(xs)-0.5,max(xs)+2,min(ys)-1,max(ys)+1))
    #adjust_text(texts)

    plt.show()
