import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'jinwon'

from matplotlib import pyplot

plt.style.use('ggplot')

class Visualizer():

    def __init__(self):
        self.predict = None
        self.true = None
        self.col_name = ['Px2', 'Px3', 'Px4', 'Py2', 'Py3', 'Py4', 'Pz2', 'Pz3', 'Pz4']
        self.data_type = None
        self.data_description = dict()

    def load_data(self, load_dir, data_type):
        print 'loading files'
        predict = dict()
        true = dict()
        self.data_type = data_type
        for dirpath, directory, files in os.walk(os.path.join(load_dir, data_type)):
            for filename in files:
                match_pred = re.match('{0}_(.+,.+,.+)_pred\.npy'.format(data_type), filename)
                if match_pred != None:
                    key = match_pred.groups()
                    predict[key] = np.load(os.path.join(dirpath, filename))


                match_true = re.match('{0}_(.+,.+,.+)_true.npy'.format(data_type), filename)
                if match_true != None:
                    key = match_true.groups()
                    true[key] = np.load(os.path.join(dirpath, filename))

        with open(os.path.join(load_dir, 'data_description.txt')) as f:
            for line in f.readlines():
                key, val = re.match('(.*) : (.*)', line).groups()
                self.data_description[key] = val

        self.predict = predict
        self.true = true

    def plot_2d(self, sensor_id=None):
        print 'start 2d ploting'
        stride = int(self.data_description['stride'])
        for key in self.predict.keys():
            X = self.predict[key]
            y = self.true[key]
            window = np.arange(X.shape[1])
            window_size = X.shape[1]
            if sensor_id == None:
                sensor_list = np.arange(X.shape[2])
            else:
                sensor_list = [int(self.col_name.index(x)) for x in sensor_id]
            for sensor_num in sensor_list:
                for timestep in np.arange(X.shape[0]):
                    plt.plot(window + timestep, X[timestep, :, sensor_num])
                    plt.plot(window +timestep, y[timestep, :, sensor_num])
                    plt.xlabel('time window from {0} to {1}'.format(stride * timestep, stride * timestep + window_size))
                    plt.ylabel(self.col_name[sensor_num])
                    plt.legend(loc = 'upper left')
                    plt.show()
                    skip = raw_input('to skip go to next columns type skip : ')
                    if skip == 'skip':
                        break

    def plot_3d(self, sensor_num=None):
        print 'start 3d ploting'
        stride = int(self.data_description['stride'])
        for key in self.predict.keys():
            predict = self.predict[key]
            true = self.true[key]

            window_size = predict.shape[1]
            sensor_list = sorted([self.col_name.index(x) for x in self.col_name if re.match('P.{0}'.format(sensor_num),x)])

            for timestep in np.arange(predict.shape[0]):
                fig = plt.figure()
                ax = fig.gca(projection = '3d')
                ax.plot(predict[timestep, :, sensor_list[0]], predict[timestep,:,sensor_list[1]], predict[timestep,:,sensor_list[2]], label = 'prediction')
                ax.plot(true[timestep,:,sensor_list[0]], true[timestep,:,sensor_list[1]], true[timestep,:,sensor_list[2]], label = 'true')
                plt.suptitle('data : {0} sensor : P{1}'.format(key, sensor_num), fontweight = 'bold')
                plt.title('time window from {0} to {1}'.format(stride * timestep, stride * timestep + window_size))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                plt.legend(loc = 'upper right')
                plt.show()

    def error_plot(self, load_dir):
        train_loss = pd.read_csv(os.path.join(load_dir, 'train_loss.csv'))
        test_loss = pd.read_csv(os.path.join(load_dir, 'test_loss.csv'))

        plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
        plt.plot(np.arange(len(train_loss)), test_loss, label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss by epoch')
        plt.legend(loc='upper right')
        plt.show()







