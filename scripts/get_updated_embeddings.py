from keras.models import Sequential,model_from_json
from alphabet import Alphabet
import cPickle
from tqdm import tqdm

distant_model_path = 'model/filtered_cleaned_160M/distant_phase_model.json'
distant_weight_path = 'model/filtered_cleaned_160M/distant_phase_weights.h5'
vocab_fname = 'preprocessed_data/vocab_mixed_2Mwords.pickle'

print('Load Distant Model')
model = model_from_json(open(distant_model_path).read())
model.load_weights(distant_weight_path)

embeddings = model.layers[0].get_weights()[0]
print embeddings.shape

print ('Load Alphabet')
alphabet = cPickle.load(open(vocab_fname))
print "alphabet", len(alphabet)

inv_alphabet = {v[0]: k for k, v in alphabet.items()}
ofile = open('embeddings/smiley_tweets_embedding_multilingual300M_updated','w')
ofile.write('{} {}\n'.format(embeddings.shape[0],embeddings.shape[1]))

for i in tqdm(xrange(len(alphabet))):
    word = inv_alphabet.get(i)
    vec = map(str,embeddings[i])
    vector = ' '.join(vec)
    line = '{} {}\n'.format(word,vector)
    ofile.write(line)

ofile.close()