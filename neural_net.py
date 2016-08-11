from __future__ import print_function
import numpy as np
import os
import getopt
import sys
from data_utils import load_supervised,DistantDataIterator,pop_layer
from evaluation_metrics import semeval_f1_score,semeval_f1_taskA
np.random.seed(1337)  # for reproducibility

from keras.callbacks import EarlyStopping,ModelCheckpoint,RemoteMonitor
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation,ZeroPadding1D,Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D,Flatten
from keras.optimizers import SGD,Adadelta
from keras import backend as K
from keras.utils.np_utils import to_categorical,probas_to_classes
from custom_callback import F1History
from sklearn.metrics import f1_score

data_dir = 'preprocessed_data'
embedding_fname = 'emb_smiley_tweets_embedding_multilingual300M.npy'
data_dir_appendix = ''
save_distant_model = True
load_distant_model = False
load_supervised_model = False
language = 'multi'
n_samples = int(open('misc/{}/nTweets.txt'.format(data_dir_appendix), 'rb').read())
sample_label = ''
random_wemb = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:e:lm:s:wr", ["dir=","embedding=","load=","language=","samples=","load-supervised=","random_emb="])
except getopt.GetoptError:
    print('Bad Input Args')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-d", "--dir"):
        data_dir_appendix = arg
    elif opt == '-l':
        save_distant_model = False
        load_distant_model = True
    elif opt == '-w':
        save_distant_model = False
        load_distant_model = True
        load_supervised_model = True
    elif opt in ("-e", "--embedding"):
        embedding_fname = '{}.npy'.format(arg)
    elif opt in ("-m", "--language"):
        language = arg
    elif opt in ("-s", "--samples"):
        if arg == 'all':
            n_samples = int(open('misc/{}/nTweets.txt'.format(data_dir_appendix), 'rb').read())
        else:
            n_samples = int(arg)
        sample_label = '_{}'.format(int(n_samples / 1000000))
    elif opt == '-r':
        random_wemb = True


n_samples = int(open('misc/{}/nTweets.txt'.format(data_dir_appendix), 'rb').read())
data_dir = 'preprocessed_data_{}'.format(data_dir_appendix)
print(data_dir)

model_dir = 'model/{}'.format(data_dir_appendix)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_dir = 'model/{}/{}'.format(language, data_dir_appendix)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

res_dir = 'results/{}/{}'.format(language, data_dir_appendix)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

distant_model_path =   'model/{}/distant_phase_model.json'.format(data_dir_appendix)
distant_weight_path =  'model/{}/distant_phase_weights{}.h5'.format(data_dir_appendix,sample_label)
super_model_path =     'model/{}/{}/super_phase_model.json'.format(language, data_dir_appendix)
super_weight_path =    'model/{}/{}/super_phase_weights{}.h5'.format(language, data_dir_appendix,sample_label)

# set parameters:
maxlen = 140
batch_size = 1000
embedding_dims = 52
nb_filter = 200
filter_length = 6
filter_height = 40
hidden_dims = nb_filter
nb_epoch = 2
n_classes_distant = 2
n_classes_super = 3


print(n_samples)

print('Loading Embeddings...')

fname_wordembeddings = os.path.join('preprocessed_data', embedding_fname)
vocab_emb = np.load(fname_wordembeddings)
print("Word embedding matrix size:", vocab_emb.shape)
max_features = vocab_emb.shape[0]

(Tids_train,X_train,y_train),(Tids_valid,X_valid,y_valid),test_sets = load_supervised(data_dir,lang=language)

datait = DistantDataIterator(data_dir,'smiley_tweets.tweets.npy','smiley_tweets.sentiments.npy')

##Layers
print('Build Model...')


if not random_wemb:
    print('Pretrained Word Embeddings')
    embeddings = Embedding(
        max_features,
        embedding_dims,
        input_length=maxlen,
        weights=[vocab_emb],
        dropout=0.2,
    )
else:
    print('Random Word Embeddings')
    embeddings = Embedding(max_features, embedding_dims,init='lecun_uniform',input_length=maxlen,dropout=0.2)

zeropadding = ZeroPadding1D(filter_length-1)


conv1 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1)

max_pooling1 = MaxPooling1D(pool_length=4,stride=2)

def max_1d(X):
    return K.max(X, axis=1)

conv2 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1)

max_pooling2 = MaxPooling1D(pool_length=2)

conv3 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1)

max_pooling3 = MaxPooling1D(pool_length=2)

conv4 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1)

max_pooling4 = MaxPooling1D(pool_length=2)

conv5 = Convolution1D(
    nb_filter=nb_filter,
    filter_length=filter_length,
    border_mode='valid',
    activation='relu',
    subsample_length=1)


hidden1 = Dense(hidden_dims)
hidden2 = Dense(hidden_dims)
hidden3 = Dense(hidden_dims)

model = Sequential()
model.add(embeddings)
model.add(zeropadding)
model.add(conv1)
model.add(max_pooling1)
model.add(conv2)
#model.add(max_pooling2)
#model.add(conv3)
#model.add(max_pooling3)
#model.add(conv4)
#model.add(max_pooling4)
#model.add(conv5)
max_pooling5 = MaxPooling1D(pool_length=model.layers[-1].output_shape[1])
model.add(max_pooling5)
model.add(Flatten())
#model.add(hidden1)
#model.add(hidden2)
model.add(hidden3)
model.add(Dropout(0.2))
#distant supervised phase
model.add(Activation('relu'))
model.add(Dense(2,activation='softmax'))
model.summary()
adadelta = Adadelta(lr=1.0,rho=0.95,epsilon=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=adadelta, metrics=['accuracy'])

if not load_distant_model:
    model.fit_generator(datait.flow(1500),samples_per_epoch=n_samples,nb_epoch=1,verbose=1)

if save_distant_model:
    print('Storing Distant Model')
    json_string = model.to_json()
    open(distant_model_path,'w+').write(json_string)
    model.save_weights(distant_weight_path,overwrite=True)

if load_distant_model:
    print('Load Distant Model')
    model = model_from_json(open(distant_model_path).read())
    model.load_weights(distant_weight_path)

pop_layer(model)
model.add(Dense(3,activation='softmax'))

model.summary()
adadelta = Adadelta(lr=1.0,rho=0.95,epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=[semeval_f1_score])

early_stop = EarlyStopping(monitor='val_semeval_f1_score',patience=50,verbose=1,mode='max')
model_checkpoit = ModelCheckpoint(filepath=super_weight_path,verbose=1,save_best_only=True,monitor='val_semeval_f1_score',mode='max')
remote_moitor = RemoteMonitor(root='http://localhost:9000')
acc_history = F1History()

output = ''
if not load_supervised_model:
    hist = model.fit(X_train,to_categorical(y_train,3), batch_size=1000,nb_epoch=1000,validation_data=(X_valid,to_categorical(y_valid,3)),callbacks=[early_stop,model_checkpoit,acc_history])
    json_string = model.to_json()
    open(super_model_path, 'w').write(json_string)
    np.save(open(os.path.join(res_dir, 'f1_epoch_history'), 'w'), np.array(acc_history.f1_scores_epoch))
    np.save(open(os.path.join(res_dir, 'f1_training_history'), 'w'), np.array(acc_history.train_score))
    max_score = max(acc_history.f1_scores_epoch)

    output = '{}\t{}\t{}: {}\t'.format(data_dir_appendix, language, 'Max Score', max_score)

print('Load Supervised  Model')
model.load_weights(super_weight_path)


for tids,X_test,y_test,name in test_sets:
    raw_data = open(os.path.join('semeval', '{}.tsv'.format(name)), 'r').readlines()
    raw_data = map(lambda x: x.replace('\n', '').split('\t'), raw_data)
    raw_tweets = map(lambda x: (x[0],x[-1]), raw_data)
    raw_lables = map(lambda x:(x[0],x[-2]),raw_data)

    raw_data_dict = dict(raw_tweets)
    raw_lables_dict = dict(raw_lables)
    ofile = open(os.path.join(res_dir, name),'w')

    y_pred = model.predict(X_test)
    y_pred = probas_to_classes(y_pred)
    score = semeval_f1_taskA(y_test,y_pred)
    scores = f1_score(y_test,y_pred,average=None)
    output += '{}: {}\t'.format(name,score)
    output += 'neg_f1: {}\t neut_f1: {}\t pos_f1: {}'.format(scores[0],scores[1],scores[2])

    for tid,label in zip(tids,y_pred):
        tweet = raw_data_dict[tid].replace('\n','')
        truth = raw_lables_dict[tid]
        l = {0:'negative',1:'neutral',2:'positive'}.get(label)
        outline = '{}\t{}\t{}\n'.format(tweet,truth,l)
        ofile.write(outline)

open(os.path.join('results','results_log.tsv'),'a').write(output + '\n')