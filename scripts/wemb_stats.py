from gensim.models import Word2Vec
model = Word2Vec.load_word2vec_format('smiley_tweets_embedding_multilingual300M_clean', binary=False)
print 'Model Loaded'

