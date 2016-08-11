import numpy as np
from matplotlib import pyplot as plt
import os

dirs = ['de','it','nl','en','multi']
training = 'f1_training_history'
test = 'f1_epoch_history'


for lang in dirs:
    dir = '../results/{}'.format(lang)
    listing = map(lambda x: os.path.join(dir,x),os.listdir(dir))
    for d in listing:
        y_train = np.load(d + '/' + training)
        y_test = np.load(d + '/' + test)

        plt.plot(y_train)
        plt.plot(y_test)
        plt.show()