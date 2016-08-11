from utils import load_glove_vocabulary

vocab = load_glove_vocabulary('../embeddings/smiley_tweets_embedding_multilingual500M10T',' ')
print len(vocab)