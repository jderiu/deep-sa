import keras


class F1History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f1_scores_epoch = []
        self.train_score = []

    def on_epoch_end(self, epoch, logs={}):
        self.f1_scores_epoch.append(logs.get('val_semeval_f1_score'))
        self.train_score.append(logs.get('semeval_f1_score'))