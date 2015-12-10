from eeg_GAL import GAL_data

__author__ = 'jinwon'

from sklearn import svm, lda, qda
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class other_classifer():

    def __init__(self, classifier):
        if classifier == 'svm':
            self.clf = svm.SCV()
        elif classifier == 'lda':
            self.clf = lda.LDA()
        elif classifier == 'qda':
            self.clf = qda.QDA()

    def train_svm(self, X, y):
        self.clf.fit(X, y)

    def evaluate(self, X, y):
        pred = self.clf.predict(X)

        accuracy = accuracy_score(y, pred)

        cm = confusion_matrix(y, pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return accuracy, cm_normalized



nb_epoch = 100
participator = 1
gal = GAL_data()
gal.load_data(load_list=['eeg', 'info'])
data_description = gal.get_data_description()
event_list=['Idle', 'Recognition_Phase', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']

data_len=gal.part_data_count[participator]
data_split_ratio = [0.8, 0.1, 0.1]
train_list = np.arange(int(data_len * data_split_ratio[0]))
validate_list = np.arange(int(data_len * data_split_ratio[1]))
test_list = np.arange(data_len - int(data_len * data_split_ratio[0]) - int(data_len * data_split_ratio[1]))

for epoch in range(nb_epoch):
    generator = gal.data_generator_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)

    def run(data_list):
        for _ in data_list:
            X , y_temp = generator.next()
            pred_temp = np.expand_dims(, axis=1)

            if pred == None:
                pred = pred_temp
                y = np.zeros((y_temp.shape[0],1))
                for i in range(y_temp.shape[1]):
                    y[np.where(y_temp[:,i] == 1)] = i
            else:

                pred = np.vstack((pred, pred_temp))

                y_temp_categorize = np.zeros((y_temp.shape[0], 1))
                for i in range(y_temp.shape[1]):
                    y_temp_categorize[np.where(y_temp[:,i] == 1)] = i
                y = np.vstack((y, y_temp_categorize))

    start = time.clock()
    rnn.run_model_with_generator_event_classify(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
    logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))