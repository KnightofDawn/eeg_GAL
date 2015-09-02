import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
__author__ = 'jinwon'

from matplotlib import pyplot



class Visualizer():

    def __init__(self):
        self.X = None
        self.y = None
        self.col_name = ['Px1', 'Px2', 'Px3', 'Px4', 'Py1', 'Py2', 'Py3', 'Py4', 'Pz1', 'Pz2', 'Pz3', 'Pz4']
        self.data_type = None
        self.data_description = dict()




    def load_data(self, load_dir, data_type):
        print 'loading files'
        X = dict()
        y = dict()
        self.data_type = data_type
        for dirpath, directory, files in os.walk(os.path.join(load_dir, data_type)):
            for filename in files:
                match_pred = re.match('train_(.+,.+,.+)_pred\.npy', filename)
                if match_pred != None:
                    key = match_pred.groups()
                    X[key] = np.load(os.path.join(dirpath, filename))


                match_true = re.match('train_(.+,.+,.+)_true.npy', filename)
                if match_true != None:
                    key = match_true.groups()
                    y[key] = np.load(os.path.join(dirpath, filename))

        with open(os.path.join(load_dir, 'data_description.txt')) as f:
            for line in f.readlines():
                key, val = re.match('(.*) : (.*)', line).groups()
                self.data_description[key] = val

        self.X = X
        self.y = y




    def make_plot(self, sensor_id=None):
        print 'start ploting'
        stride = int(self.data_description['stride'])
        for key in self.X.keys():
            X = self.X[key]
            y = self.y[key]
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


data_dir = '2015-09-01 21:27:15.623976'
load_dir = os.path.join('output', data_dir)

vis = Visualizer()
sensor_id = ['Px1', 'Py1', 'Pz1']
vis.load_data(load_dir = load_dir, data_type='train')
vis.make_plot(sensor_id = sensor_id)



