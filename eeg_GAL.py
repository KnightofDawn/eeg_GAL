# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import pdb
import re
import cPickle
import datetime

import pandas as pd
from scipy.io import loadmat
import numpy as np
import logging
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleDeepRNN
import time
import sklearn

from sklearn.metrics import roc_auc_score

__author__ = 'jinwon'



class GAL_data():

    def __init__(self):
        self.eeg_dict = None
        self.info_dict = None

        self.total_count = None
        self.trial_length = None

        self.timesteps = None
        self.X_dict = None
        self.y_dict = None

        self.part_data_count = dict.fromkeys(np.arange(12) + 1, 0)
        self.data_description = dict()

    def set_logger(self, logger):
        self.logger = logger
        
    def read_raw_data(self):
        data_dir = u'//media//jinwon//새 볼륨//Downloads//grasp_and_lift_eeg'
        eeg_dict = dict()
        emg_dict = dict()
        kin_dict = dict()
        info_dict = dict()

        col_name = \
        ['Part',
        'Run',
        'Lift',
        'CurW',
        'CurS',
        'PrevW',
        'PrevS',
        'StartTime',
        'LEDOn',
        'LEDOff',
        'BlockType',
        'tIndTouch',
        'tThumbTouch',
        'tFirstDigitTouch',
        'tBothDigitTouch',
        'tIndStartLoadPhase',
        'tThuStartLoadPhase',
        'tBothStartLoadPhase',
        'tLiftOff',
        'tReplace',
        'tIndRelease',
        'tThuRelease',
        'tBothReleased',
        'GF_Max',
        'LF_Max',
        'dGF_Max',
        'dLF_Max',
        'tGF_Max',
        'tLF_Max',
        'tdGF_Max',
        'tdLF_Max',
        'GF_Hold',
        'LF_Hold',
        'tHandStart',
        'tHandStop',
        'tPeakVelHandReach',
        'tPeakVelHandRetract',
        'GripAparture_Max',
        'tGripAparture_Max',
        'Dur_Reach',
        'Dur_Preload',
        'Dur_LoadPhase',
        'Dur_Release']


        for datapath, directory, filelist in os.walk(data_dir):
            for file in filelist:
                info_match = re.match('P\d+_info.mat', file)
                if info_match != None:
                    info_mat = loadmat(os.path.join(datapath, file))
                    part_num = [int(x) for x in info_mat['info'][:,0]]
                    series_num = [int(x) for x in info_mat['info'][:,1]]
                    trial_num = [int(x) for x in info_mat['info'][:,2]]
                    row_num = np.arange(info_mat['info'].shape[0])
                    for p, s, t, pos in zip(part_num, series_num, trial_num, row_num):
                        data = info_mat['info'][pos,3:].reshape(1,40)
                        info_dict[(p,s,t)] = pd.DataFrame(data=data, columns=col_name[3:])

                eeg_match = re.match('WS_P(\d+)_S(\d+)_T(\d+)_eeg.mat', file)

                if eeg_match != None:
                    try :
                        eeg_mat = loadmat(os.path.join(datapath, file))
                        key = tuple([int(x) for x in eeg_match.groups()])
                        eeg_dict[key] = eeg_mat['eeg']
                    except ValueError:
                        self.logger.info( file, 'ValueError')


        self.eeg_dict = eeg_dict
        self.info_dict = info_dict

        self.logger.info( 'read raw data')

        self.count_data()

    def save_raw_data(self):
        # assert not os.path.exists(os.path.join('data', 'eeg_info_dict.pkl')), 'pickled data already exist'
        assert self.eeg_dict != None, 'eeg_dict null'
        assert self.info_dict != None, 'info_dict null'

        with open(os.path.join('data', 'eeg_info_dict.pkl'), 'wb') as f:
            cPickle.dump(self.eeg_dict, f, protocol = cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.info_dict, f, protocol = cPickle.HIGHEST_PROTOCOL)
        self.logger.info( 'saved pickled data : eeg_info_dict.pkl')

    def load_data(self):
        assert os.path.exists(os.path.join('data', 'eeg_info_dict.pkl')), 'pickled data doesn\'t exist'

        with open(os.path.join('data', 'eeg_info_dict.pkl'), 'rb') as f:
            self.eeg_dict = cPickle.load(f)
            self.info_dict = cPickle.load(f)
        self.logger.info( 'loaded pickled data : eeg_info_dict.pkl')

        self.count_data()

    def count_data(self):
        for key in self.eeg_dict.keys():
            part = key[0]
            self.part_data_count[part] = self.part_data_count[part] + 1

    def examine_time(self):
        assert self.info_dict != None
        on_stop_time = list()
        on_off_time = list()
        led_on_time = list()
        led_off_time = list()
        hand_stop_time = list()
        for key in self.info_dict:
            led_on = self.info_dict[key]['LEDOn']
            led_off = self.info_dict[key]['LEDOff']
            hand_stop = self.info_dict[key]['tHandStop']
            data_len = self.eeg_dict[key].shape[0]

            self.logger.info( 'data len : ', data_len)
            self.logger.info( 'handstop : ', hand_stop/0.002)

            on_stop_time.extend(hand_stop - led_on)
            on_off_time.extend(led_off - led_on)
            led_on_time.extend(led_on)
            led_off_time.extend(led_off)
            hand_stop_time.extend(hand_stop)

        col_list = [on_stop_time, on_off_time, led_on_time, led_off_time, hand_stop_time]
        col_name_list = ['on_stop_time', 'on_off_time', 'led_on_time', 'led_off_time', 'hand_stop_time']
        df = pd.DataFrame(columns = ['min', '20', 'median','95','97','99', 'max'])

        for val, name in zip(col_list, col_name_list):
            row_val = (min(val), np.percentile(val, 20),np.percentile(val, 50), np.percentile(val, 95), np.percentile(val, 97), np.percentile(val, 99), max(val))
            row_val =[val/0.002 for val in row_val[:]]
            df.loc[name,:] = row_val

        df.to_csv(os.path.join('result', 'time_data.csv'))
        self.logger.info( 'saved result/time_data.csv')

    def uniform_trial_length(self, trail_length):
        '''
        :param trail_length: timestep that has to same for each batch
        :return: none
        '''

        '''
            if there are not enough data points, used the 1000 points before LEDOn
            for this, used the same slice for every patient
        '''
        assert self.eeg_dict != None

        add_count = 0
        discard_count = 0
        delete_count = 0
        normal_count = 0

        led_on_step = 1000

        for key in self.eeg_dict.keys():
            temp = self.eeg_dict[key]
            if temp.shape[0] < trail_length:
                add_count = add_count + 1
                add_amount = trail_length - temp.shape[0]

                # if add_amount < led_on_step:
                #     add = temp[:add_amount]
                #     temp = np.vstack((add, temp))  #add to the front
                #     assert temp.shape[0] == timesteps
                # else:
                #     # hand_stop_time = self.info_dict[key]['tHandStop']
                #     # after_stop_time = hand_stop_time - temp.shape[0]
                #     # if after_stop_time + led_on_step > timesteps:           #add to front(before led_on) and end ( after handstop)
                #     #     add_left_amount = timesteps - temp.shape[0] - led_on_step
                #     #     temp = np.vstack((temp[:led_on_step], temp, temp[hand_stop_time:hand_stop_time+add_left_amount]))
                #     # assert temp.shape[0] == timesteps
                #     add = temp[:add_amount]
                #     left_add_amount = int(add_amount - led_on_step)
                #     quotient = left_add_amount / led_on_step
                #     remainder = left_add_amount % led_on_step
                #
                #     if quotient == 0:
                #         add = temp[:remainder]
                #     else:
                #         for i in range(quotient):
                #             if i == 0:
                #                 add = temp[:add_amount]
                #             else:
                #                 add = np.vstack((add, temp[:add_amount]))
                #         add = np.vstack((add, temp[:remainder]))
                #
                #     temp = np.vstack((add, temp))
                #     assert temp.shape[0] == timesteps

                quotient = add_amount / led_on_step
                remainder = add_amount % led_on_step

                if quotient == 0:
                    add = temp[:remainder]
                else:
                    for i in range(quotient):
                        if i == 0:
                            add = temp[:led_on_step]
                        else:
                            add = np.vstack((add, temp[:led_on_step]))
                    add = np.vstack((add, temp[:remainder]))

                temp = np.vstack((add, temp))
                assert temp.shape[0] == trail_length


            elif temp.shape[0] > trail_length:
                discard_count = discard_count + 1
                discard_amount = temp.shape[0] - trail_length
                if discard_amount <= led_on_step:
                    temp = temp[discard_amount:]
                    assert temp.shape[0] == trail_length
                else:
                    stop_step = self.info_dict[key]['tHandStop'].values[0] / 0.002
                    after_stop_step = temp.shape[0] - stop_step
                    if discard_amount < led_on_step + after_stop_step:
                        discard_left_amount = discard_amount - led_on_step
                        temp = temp[led_on_step:-discard_left_amount]
                        assert temp.shape[0] == trail_length
                    else:
                        temp = None
                        discard_count = discard_count - 1
                        delete_count = delete_count + 1
            else:
                normal_count = normal_count + 1
                pass # temp.shape[0] == timesteps necessarily not needed

            if temp != None:
                self.eeg_dict[key] = temp
            else:
                self.eeg_dict.pop(key)

        self.logger.info( 'discarded count :', discard_count)
        self.logger.info( 'added count : ', add_count)
        self.logger.info( 'deleted count :', delete_count)
        self.logger.info( 'total count :', normal_count + add_count + discard_count)

        self.total_count = normal_count + add_count + discard_count
        self.trial_length = trail_length

    def X_y_part_generator(self, part, timesteps, stride, event_list, event_range=None, input_dim=32):

        if event_range != None:
            assert type(event_range) == np.ndarray, 'event_range not numpy array'

        assert input_dim == self.eeg_dict[self.eeg_dict.keys()[0]].shape[1], 'input dim doesn\'t match'

        self.data_description['participator'] = part

        for key in self.data_description.keys():
            if self.data_description[key] == None:
                self.data_description['timesteps'] = timesteps
                self.data_description['stride'] = stride
                self.data_description['event_list'] = event_list
            if event_range != None:
                self.data_description['event_range'] = event_range

        part_list = [x for x in self.eeg_dict.keys() if x[0] == part]
        for key in part_list:
            # nb_samples = (self.trial_length - timesteps) / stride + 1
            trial_step_size = self.eeg_dict[key].shape[0]
            nb_samples = (trial_step_size - timesteps) / stride + 1
            X = np.zeros((nb_samples, timesteps, input_dim))
            y = np.zeros((nb_samples, len(event_list)))
            temp = self.eeg_dict[key]

            for i in np.arange(nb_samples):
                X[i,:,:] = temp[i*stride:i*stride+timesteps,:]

            if event_range != None:
                for index, event in enumerate(event_list):
                    event_time_step = int(self.info_dict[key][event].values / 0.002)
                    y[(event_time_step + event_range - trial_step_size)/stride, index] = 1
            else:
                handstart = int(self.info_dict[key]['tHandStart'].values / 0.002)
                bothdigittouch = int(self.info_dict[key]['tBothDigitTouch'].values / 0.002)
                bothstartload = int(self.info_dict[key]['tBothStartLoadPhase'].values / 0.002)
                liftoff = int(self.info_dict[key]['tLiftOff'].values / 0.002)
                replace = int(self.info_dict[key]['tReplace'].values / 0.002)
                bothreleased = int(self.info_dict[key]['tBothReleased'].values / 0.002)
                handstop = int(self.info_dict[key]['tHandStop'].values / 0.002)

                y[handstart:bothdigittouch, event_list.index('Dur_Reach')] = 1
                #y[bothdigittouch:bothstartload, event_list.index('Dur_Preload')] = 1
                y[bothdigittouch:liftoff, event_list.index('Dur_LoadPhase')] = 1
                '''
                Dur_LoadPhase is different from the definition of the original paper
                '''
                y[replace:bothreleased, event_list.index('Dur_Release')] = 1
                y[bothreleased:handstop, event_list.index('Dur_Withdraw')] = 1

            X = X.astype('float32')
            y = y.astype('float32')

            yield [X, y]

    def save_X_y(self):
        for key in self.X_dict.keys():
            np.save(os.path.join('data', 'X_'+str(key)+'.npy'), self.X_dict[key])
            np.save(os.path.join('data', 'y_'+str(key)+'.npy'), self.y_dict[key])
        self.logger.info( 'saved X and y')

    def get_data_description(self):
        return self.data_description

    def set_logger(self, logger):
        self.logger=logger

class EEG_rnn_batch():

    def __init__(self, event_list):
        self.X_dict = None
        self.y_dict = None
        self.event_list = event_list
        self.model_config = dict()
        self.data_description = None

    def load_data(self):

        self.X_dict = dict()
        self.y_dict = dict()

        for file_name in os.listdir(os.path.join('data')):
            match_X = re.match('X_((\d+), (\d+), (\d+)).npy', file_name)
            match_y = re.match('y_((\d+), (\d+), (\d+)).npy', file_name)

            if match_X != None:
                key = tuple([int(x) for x in match_X.groups()])
                self.X_dict[key] = np.load(os.path.join('data', file_name))

            elif match_y != None:
                key = tuple([int(x) for x in match_X.groups()])
                self.y_dict[key] = np.load(os.path.join('data', file_name))

    def select_model(self, model_name):

        def lstm_1():
            model = Sequential()
            model.add(LSTM(input_dim=32, output_dim=64)) # try using a GRU instead, for fun
            # model.add(Dropout(0.5))
            model.add(Dense(64, len(self.event_list)))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adadelta', class_mode='binary')
            return model

        def lstm_2():
            model = Sequential()
            model.add(LSTM(input_dim=32, output_dim=64, return_sequences=True)) # try using a GRU instead, for fun
            model.add(LSTM(input_dim=64, output_dim=64, return_sequences=False))
            model.add(Dense(64, len(self.event_list)))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adadelta', class_mode='binary')
            return model

        def simple_rnn_1():
            model = Sequential()
            model.add(SimpleDeepRNN(input_dim=32, output_dim=64)) # try using a GRU instead, for fun
            # model.add(Dropout(0.5))
            model.add(Dense(64, len(self.event_list)))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adadelta', class_mode='binary')
            return model


        self.model = eval('{0}()'.format(model_name))
        self.model_config['model'] = model_name
        self.logger.info('selected model {0}'.format(model_name))

    def load_model_weight(self, model_name, folder_name):
        self.model.load_weights(os.path.join('output', folder_name, 'weight.hdf'))
        self.model_config['weight_loaded_from'] = folder_name
        self.logger.info('loaded model weight from {0}'.format(folder_name))

    def run_model_temp(self, part_list, batch_size, nb_epoch, train_test_ratio):

        self.model_config['partcipant_list'] = part_list
        self.model_config['batch size'] = batch_size
        self.model_config['number of epochs'] = nb_epoch
        self.model_config['number of epochs'] = train_test_ratio

        for part in part_list:
            key_list = [x for x in self.X_dict.keys() if x == part]
            train_num = int(len(key_list) * train_test_ratio)
            train_list = key_list[:train_num]
            test_list = key_list[train_num:]

            for epoch in range(nb_epoch):
                self.logger.info( 'epoch', epoch)
                for key in train_list:
                    X_train = self.X_dict[key]
                    y_train = self.y_dict[key]
                    loss, accuracy = self.model.train_on_batch(X = X_train, y = y_train, accuracy = True)
                    self.logger.info( 'train key : {0}, loss : {1}, accuracy : {2}'.format(key, loss, accuracy))

                for key in test_list:
                    X_test = self.X_dict[key]
                    y_test = self.y_dict[key]
                    loss, accuracy = self.model.test_on_batch(X = X_test, y = y_test, accuracy = True)
                    self.logger.info( 'test key : {0}, loss : {1}, accuracy : {2}'.format(key, loss, accuracy))

    def run_model_with_generator(self, generator, train_list, validate_list, test_list):

        self.model_config['train_data_size'] = len(train_list)
        self.model_config['validate_data_size'] = len(validate_list)
        self.model_config['test_data_size'] = len(test_list)

        def run(data_list, data_type, generator=generator):
            loss_total = 0
            accuracy_total = 0
            auc_total = 0
            for i in data_list:
                X, y = generator.next()
                loss, accuracy = self.model.train_on_batch(X = X, y = y, accuracy = True)
                loss_total = loss_total + loss
                accuracy_total = accuracy_total + accuracy
                if data_type == 'test' or data_type == 'validate':
                    pred = self.model.predict(X)
                    auc = np.asarray([roc_auc_score(y[:, j], pred[:, j]) for j in range(y.shape[1])])
                    auc_total = auc_total + auc
                    mean_auc_total = np.mean(auc_total)
                    print( '{0} : complete {1:.2f}%,  loss : {2}, auc : {3}'.format(data_type, float(i+1)/len(data_list) * 100, loss_total/(i+1), mean_auc_total/(i+1)), end = '\r')
                elif data_type == 'train':
                    print( '{0} : complete {1:.2f}%,  loss : {2}'.format(data_type, float(i+1)/len(data_list) * 100, loss_total/(i+1)), end = '\r')

            if data_type == 'test' or data_type == 'validate':
                self.logger.info( '{0} : complete {1:.2f}%,  loss : {2}, auc : {3}'.format(data_type, float(i+1)/len(data_list) * 100, loss_total/(i+1), mean_auc_total/(i+1)))

            elif data_type == 'train':
                self.logger.info( '{0} : complete {1:.2f}%,  loss : {2}'.format(data_type, float(i+1)/len(data_list) * 100, loss_total/(i+1)))

        run(data_list=train_list, data_type = 'train')
        run(data_list = validate_list, data_type='validate')
        run(data_list=test_list, data_type= 'test')

    def save_score_model(self, generator, train_list, validate_list, test_list, event_list, output_dir):
        assert self.data_description != None, 'data_description is not set'

        def evaluate_score(self, data_list, data_type, param_threshold = None, generator=generator, event_list=event_list, output_dir = output_dir):
            start = time.clock()
            self.logger.info('predicting values : {0}'.format(data_type))

            pred = None
            pred_binary = None
            for _ in data_list:
                X , y_temp = generator.next()
                if pred == None:
                    pred = self.model.predict(X)
                    pred_binary = np.zeros(shape = y_temp.shape)
                    y = y_temp
                else:
                    pred = np.vstack((pred, self.model.predict(X)))
                    pred_binary = np.vstack((pred_binary, np.zeros(shape = y_temp.shape)))
                    y = np.vstack((y, y_temp))

            if data_type == 'validate':
                threshold_list = np.arange(0.5, 0.01, -0.01)
            elif data_type == 'test' or data_type == 'train':
                threshold_list = [param_threshold]

            max_mean_f1 = 0
            max_f1_score = 0
            param_threshold = 1

            if data_type == 'validate':
                multi_index = pd.MultiIndex.from_product([threshold_list, self.event_list], names = ['threshold', 'event'])
                df_score_all = pd.DataFrame(index = multi_index)
                score_name_list = ['f1_score', 'precision_score', 'recall_score']
            elif data_type == 'test' or data_type == 'train':
                df_score_all = None

            nb_classes = len(event_list)

            for threshold in threshold_list:
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[1]):
                        if pred[i, j] >= threshold:
                            pred_binary[i, j] = 1
                f1_score = [sklearn.metrics.f1_score(y[:, i], pred_binary[:, i]) for i in range(nb_classes)]
                mean_f1 = np.mean(f1_score)
                f1_score = sklearn.metrics.f1_score(y[:, k], pred_binary[:, k])

                if data_type =='validate':
                    precision_score = [sklearn.metrics.precision_score(y[:, i], pred_binary[:, i]) for i in range(nb_classes)]
                    recall_score = [sklearn.metrics.recall_score(y[:, i], pred_binary[:, i]) for i in range(nb_classes)]
                    score_list = [f1_score, precision_score, recall_score]

                    for score, col in zip(score_name_list, range(len(score_name_list))):
                        df_score_all.loc[(threshold,), score] = score_list[col]

                if mean_f1 > max_mean_f1:
                    max_mean_f1 = mean_f1
                    max_f1_score = f1_score
                    param_threshold = threshold
                    max_precision_score = [sklearn.metrics.precision_score(y[:, i], pred_binary[:, i]) for i in range(nb_classes)]
                    max_recall_score = [sklearn.metrics.recall_score(y[:, i], pred_binary[:, i]) for i in range(nb_classes)]

            #precision, recall, thresholds = [sklearn.metrics.precision_recall_curve(y[:, i], pred[:,i]) for i in range(nb_classes)]

            auc_score = [roc_auc_score(y[:, i], pred[:, i]) for i in range(nb_classes)]

            score_list = [max_f1_score, max_precision_score, max_recall_score, auc_score]

            self.logger.info( ('time elapsed : {0} seconds'.format(time.clock() - start)))

            if data_type == 'validate':
                self.logger.info((data_type+' threshold : ', param_threshold))
                self.logger.info( (data_type+' auc : ', np.mean(auc_score)))
                self.logger.info( (data_type+' f1 : ', np.mean(max_f1_score)))
                self.logger.info( (data_type+' precision : ', np.mean(max_precision_score)))
                self.logger.info( (data_type+' recall : ', np.mean(max_recall_score)))
            else:
                self.logger.info( (data_type+' auc : ', np.mean(auc_score)))
                self.logger.info( (data_type+' f1 : ', np.mean(max_f1_score)))
                self.logger.info( (data_type+' precision : ', np.mean(max_precision_score)))
                self.logger.info( (data_type+' recall : ', np.mean(max_recall_score)))

            return score_list, param_threshold, df_score_all, pred, pred_binary

        def evaluate_score_revised(self, data_list, data_type, param_threshold_list = None, generator=generator, event_list=event_list, output_dir = output_dir):
            start = time.clock()
            self.logger.info('predicting values : {0}'.format(data_type))

            pred = None
            pred_binary = None
            for _ in data_list:
                X , y_temp = generator.next()
                if pred == None:
                    pred = self.model.predict(X)
                    y = y_temp
                else:
                    pred = np.vstack((pred, self.model.predict(X)))
                    y = np.vstack((y, y_temp))

            nb_classes = len(event_list)

            if param_threshold_list == None:
                param_threshold_list = list()

            if data_type == 'validate' or data_type == 'test':
                f1_score_list = list()
                precision_list = list()
                recall_list = list()
            elif data_type =='train':
                f1_score_list = None
                precision_list = None
                recall_list = None


            for i in range(nb_classes):
                if data_type == 'validate':
                    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y[:,i], pred[:,i], pos_label = 1)
                    f1_score = np.asarray([2 * prec * rec / (prec + rec) for prec, rec in zip(precision, recall)])
                    max_f1_score = max(f1_score)
                    index = np.where(f1_score == max_f1_score)
                    param_threshold_list.append(thresholds[index])
                    f1_score_list.append(max_f1_score)
                    precision_list.append(precision[index])
                    recall_list.append(recall[index])
                elif data_type == 'test':
                    pred_binary_one_class = (pred[:,i] > param_threshold_list[i]).astype(int)
                    f1_score = sklearn.metrics.f1_score(y[:,i], pred_binary_one_class)
                    precision = sklearn.metrics.precision_score(y[:,i], pred_binary_one_class)
                    recall = sklearn.metrics.recall_score(y[:,i], pred_binary_one_class)
                    f1_score_list.append(f1_score)
                    precision_list.append(precision)
                    recall_list.append(recall)

            auc_roc_score = [sklearn.metrics.roc_auc_score(y[:, i], pred[:, i]) for i in range(nb_classes)]
            auc_pr_score = [sklearn.metrics.average_precision_score(y[:, i], pred[:, i]) for i in range(nb_classes)]

            score = pd.DataFrame(index = event_list)
            score['auc_roc_score'] = auc_roc_score
            score['auc_pr_score'] = auc_pr_score
            score['f1_score'] = f1_score_list
            score['precision_score'] = precision_list
            score['recall_score'] = recall_list


            self.logger.info( ('time elapsed : {0} seconds'.format(time.clock() - start)))
            self.logger.info( (data_type+' auc_roc : ', auc_roc_score))
            self.logger.info( (data_type+' auc_pr : ', auc_pr_score))

            if data_type == 'validate':
                self.logger.info((data_type+' threshold : ', param_threshold_list))
                self.logger.info( (data_type+' max f1 : ', f1_score_list))
                self.logger.info( (data_type+' precision : ', precision_list))
                self.logger.info( (data_type+' recall : ', recall_list))
            elif data_type == 'test':
                self.logger.info( (data_type+' max f1 : ', f1_score_list))
                self.logger.info( (data_type+' precision : ', precision_list))
                self.logger.info( (data_type+' recall : ', recall_list))
            elif data_type == 'train':
                pass

            return score, param_threshold_list

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        validate_score, param_threshold_list = evaluate_score_revised(self, validate_list, 'validate')
        train_score, _ = evaluate_score_revised(self, train_list, 'train')
        test_score, _ = evaluate_score_revised(self, test_list, 'test', param_threshold_list=param_threshold_list)

        validate_score.to_csv(os.path.join(output_dir, 'validate_score.csv'))
        train_score.to_csv(os.path.join(output_dir, 'train_score.csv'))
        test_score.to_csv(os.path.join(output_dir, 'test_score.csv'))

        self.model.save_weights(os.path.join(output_dir, 'weight.hdf'))

        with open(os.path.join(output_dir, 'data_description.txt'), 'w') as f:
            for key in self.data_description.keys():
                f.write('{0} : {1}\n'.format(key, self.data_description[key]))

        self.logger.info('data_description.txt saved')

        with open(os.path.join(output_dir, 'model_config.txt'), 'w') as f:
            for key in self.model_config.keys():
                f.write('{0} : {1}\n'.format(key, self.model_config[key]))
        self.logger.info('model_config.txt saved')

    def set_data_description(self, data_description):
        self.data_description = data_description

    def set_model_config(self, key, val):
        self.model_config[key] = val

    def set_logger(self, logger):
        self.logger = logger


def read_save_raw():
    gal = GAL_data()
    gal.read_raw_data()
    gal.save_raw_data()

def check_time():
    gal = GAL_data()
    gal.load_data()
    gal.examine_time()

def legacy():
    gal = GAL_data()
    gal.load_data()
    gal.uniform_trial_length(trail_length=6000)
    event_list = ['tHandStart', 'tFirstDigitTouch', 'tBothStartLoadPhase', 'tLiftOff', 'tReplace', 'tBothReleased', 'tHandStop']
    gal.make_agg_X_y(timesteps=500, stride=1, event_list=event_list, event_range=np.arange(-75, 75))
    gal.save_X_y()


def run_model_event_range(model_name, participator, timesteps, stride, nb_epoch, event_range, load_weight_from = None):
    logger = logging.getLogger()
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data()
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))
    event_list = ['tHandStart', 'tFirstDigitTouch', 'tBothStartLoadPhase', 'tLiftOff', 'tReplace', 'tBothReleased', 'tHandStop']

    rnn = EEG_rnn_batch(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.1, 0.1]
    train_list = np.arange(int(data_len * data_split_ratio[0]))
    validate_list = np.arange(int(data_len * data_split_ratio[1]))
    test_list = np.arange(data_len - int(data_len * data_split_ratio[0]) - int(data_len * data_split_ratio[1]))

    for epoch in range(nb_epoch):
        generator = gal.X_y_part_generator(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, event_range=event_range)
        logger.info( 'epoch : {0}'.format(epoch))
        start = time.clock()
        rnn.run_model_with_generator(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))


    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.X_y_part_generator(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, event_range=event_range)

    rnn.save_score_model(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

def run_model_duration(model_name, participator, timesteps, stride, nb_epoch, load_weight_from = None):
    logger = logging.getLogger()
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data()
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))
    #event_list=['Dur_Reach', 'Dur_Preload', 'Dur_LoadPhase', 'Dur_Release', 'Dur_Withdraw']
    event_list=['Dur_Reach', 'Dur_LoadPhase', 'Dur_Release', 'Dur_Withdraw']

    rnn = EEG_rnn_batch(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.1, 0.1]
    train_list = np.arange(int(data_len * data_split_ratio[0]))
    validate_list = np.arange(int(data_len * data_split_ratio[1]))
    test_list = np.arange(data_len - int(data_len * data_split_ratio[0]) - int(data_len * data_split_ratio[1]))

    for epoch in range(nb_epoch):
        generator = gal.X_y_part_generator(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)
        logger.info( 'epoch : {0}'.format(epoch))
        start = time.clock()
        rnn.run_model_with_generator(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))


    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.X_y_part_generator(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)

    rnn.save_score_model(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

if __name__ == '__main__':
#    run_model(model_name='lstm_1', participator=1, timesteps=100, stride=1, nb_epoch=50, load_weight_from = '2015-08-24 21:10:33.270834')
#    run_model(model_name='lstm_1', participator = 1, timesteps=30, stride=1, nb_epoch=50)
#    for i in np.arange(11) + 2:
#        run_model(model_name='lstm_1', participator=i, timesteps=50, stride=1, nb_epoch=50)
#    run_model_duration(model_name='lstm_1', participator=1, timesteps=500, stride=100, nb_epoch=20, event_range=np.arange(-75, 75))
    run_model_duration(model_name='lstm_2', participator=1, timesteps=100, stride=1, nb_epoch=1, load_weight_from='2015-08-28 16:40:02.798964')