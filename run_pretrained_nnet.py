from __future__ import print_function
import numpy as np
import os
import getopt
import sys
from sklearn.metrics import accuracy_score
from evaluation_metrics import semeval_f1_score,semeval_f1_taskA
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential,model_from_json
from keras.utils.np_utils import probas_to_classes


data_dir = 'preprocessed_data_filtered_cleaned_160M_random'
embedding_fname = 'emb_smiley_tweets_embedding_multilingual300M.npy'
data_dir_appendix = ''
save_distant_model = True
load_distant_model = False
load_supervised_model = False
language = 'multi'


try:
    opts, args = getopt.getopt(sys.argv[1:], "d:m:", ["dir=","language="])
except getopt.GetoptError:
    print('Bad Input Args')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-d", "--dir"):
        data_dir_appendix = arg
    elif opt in ("-m", "--language"):
        language = arg

super_model_path = 'model/{}/{}/super_phase_model.json'.format(language, data_dir_appendix)
super_weight_path = 'model/{}/{}/super_phase_weights.h5'.format(language, data_dir_appendix)

model = model_from_json(open(super_model_path).read())

print('Load Supervised  Model')
model.load_weights(super_weight_path)

files = map(lambda x: x.replace('.tsv',''),os.listdir('random_tweets'))

for name in files:
    print(name)
    try:
        dir = os.path.join(data_dir,name)
        X_test = np.load('{}.tweets.npy'.format(dir))
        y_test = np.load('{}.sentiments.npy'.format(dir))
        y_pred = model.predict_proba(X_test)
        np.savetxt('results/{}.tsv'.format(name),y_pred,delimiter='\t')

        y_mean = map(lambda x:str(x),np.mean(y_pred,axis=0).tolist())
        y_pred = probas_to_classes(y_pred)
        score = accuracy_score(y_test, y_pred)
        output = '{}\t{}\t{}'.format(name,'\t'.join(y_mean),score)
        open(os.path.join('results', 'results_log.tsv'), 'a').write(output + '\n')
    except:
        print('Some err')



