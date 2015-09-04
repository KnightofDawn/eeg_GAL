__author__ = 'jinwon'
import logging
import os
import datetime
import numpy as np
import time
from eeg_GAL import EEG_rnn_batch, GAL_data
from Visualizer import Visualizer
import pandas as pd
import json

def read_save_raw():
    gal = GAL_data()
    gal.set_logger(logging.getLogger())
    gal.read_raw_data()
    gal.save_raw_data()

def check_time():
    gal = GAL_data()
    gal.load_data()
    gal.examine_time()

def run_model_event_range_generator(model_name, participator, timesteps, stride, nb_epoch, event_range, load_weight_from = None):
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
        rnn.run_model_with_generator_event(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))


    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.X_y_part_generator(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, event_range=event_range)

    rnn.save_event(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

def run_model_duration_generator(model_name, participator, timesteps, stride, nb_epoch, load_weight_from = None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'dur_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info'])
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))
    #event_list=['Dur_Reach', 'Dur_Preload', 'Dur_LoadPhase', 'Dur_Release', 'Dur_Retract']
    event_list=['Dur_Reach', 'Dur_LoadReach', 'Dur_LoadMaintain', 'Dur_LoadRetract', 'Dur_Retract']

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
        generator = gal.data_generator_event(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)
        logger.info( 'epoch : {0}'.format(epoch))
        start = time.clock()
        rnn.run_model_with_generator_event(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))


    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.data_generator_event(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)

    rnn.save_event_generator(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

def run_model_duration(model_name, participator, timesteps, stride, nb_epoch, load_weight_from = None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'dur_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info'])
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))
    #event_list=['Dur_Reach', 'Dur_Preload', 'Dur_LoadPhase', 'Dur_Release', 'Dur_Retract']
    event_list=['Dur_Reach', 'Dur_LoadReach', 'Dur_LoadMaintain', 'Dur_LoadRetract', 'Dur_Retract']

    rnn = EEG_rnn_batch(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_split_ratio = [0.8, 0.1, 0.1]
    data = gal.data_event(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, partition_ratio=data_split_ratio, input_dim=32)
    loss_train, loss_val, loss_test = rnn.run_model_event(data=data, nb_epoch = nb_epoch)

    with open(os.path.join(output_dir, 'train_loss.json'), 'w') as f:
        json.dump(loss_train, f)
    with open(os.path.join(output_dir, 'validate_loss.json'), 'w') as f:
        json.dump(loss_val, f)
    with open(os.path.join(output_dir, 'test_loss.json'), 'w') as f:
        json.dump(loss_test, f)

    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.data_generator_event(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)
    rnn.save_event(data=data, event_list=event_list, output_dir=output_dir)

def run_model_kin(model_name, participator, timesteps, stride, nb_epoch, load_weight_from = None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'kin_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'kin'])
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    rnn = EEG_rnn_batch(None)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.2]
    train_list = np.arange(int(data_len * data_split_ratio[0]))
    test_list = np.arange(data_len - int(data_len * data_split_ratio[0]))

    generator = gal.data_generator_kin(part=participator, timesteps=timesteps, stride=stride)
    loss_history = rnn.run_model_kin(generator=generator, train_list=train_list, test_list=test_list, nb_epoch=nb_epoch)

    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.data_generator_kin(part=participator, timesteps=timesteps, stride=stride)
    rnn.save_kin_generator(generator=generator,train_list=train_list, test_list=test_list, output_dir=output_dir)

def predict(model_name, participator, load_weight_from):
    logger_name = model_name + str(participator) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'predict_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'kin'])
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    rnn = EEG_rnn_batch(None)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.2]
    train_list = np.arange(int(data_len * data_split_ratio[0]))
    test_list = np.arange(data_len - int(data_len * data_split_ratio[0]))

    rnn.set_data_description(data_description)
    generator = gal.data_generator_kin(part=participator, timesteps=10, stride=10)
    rnn.save_kin_generator(generator=generator,train_list=train_list, test_list=test_list, output_dir=output_dir)

def run_model_kin_generator(model_name, participator, timesteps, stride, nb_epoch, patience_limit, loss_delta_limit, load_weight_from = None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'kin_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'kin'])
    gal.preprocess_kin()
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    rnn = EEG_rnn_batch(None)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data from a generator')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8,0.2]
    train_list = np.arange(int(data_len * data_split_ratio[0]))
    test_list = np.arange(data_len - int(data_len * data_split_ratio[0]))

    loss_train_df = pd.DataFrame(columns = ['epoch', 'loss'])
    loss_test_df = pd.DataFrame(columns = ['epoch', 'loss'])
    patience = 0
    for epoch in range(nb_epoch):
        generator = gal.data_generator_kin(part=participator, timesteps=timesteps, stride=stride)
        logger.info( 'epoch : {0}'.format(epoch))
        start = time.clock()
        train_loss, test_loss = rnn.run_model_with_generator_kin(generator=generator, train_list=train_list, test_list=test_list)
        loss_train_df.loc[epoch, ['epoch', 'loss']] = [epoch+1, train_loss]
        loss_test_df.loc[epoch, ['epoch', 'loss']] = [epoch+1, test_loss]
        if epoch == 0:
            prev_train_loss = train_loss
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))
        loss_delta = abs(prev_train_loss - train_loss) / prev_train_loss * 100
        if loss_delta < loss_delta_limit:
            patience = patience + 1
            if patience > patience_limit:
                logger.info('training stopped at epoch {0} due to patience threshold'.format(epoch))
                break
        else:
            patience = patience - 1

    loss_train_df.to_csv(os.path.join(output_dir, 'train_loss.csv'), index=False)
    loss_test_df.to_csv(os.path.join(output_dir, 'test_loss.csv'), index=False)

    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.data_generator_kin(part=participator, timesteps=timesteps, stride = stride)

    rnn.save_kin_generator(generator=generator,train_list=train_list, test_list=test_list, output_dir=output_dir)

def run_visualizer(data_dir, data_type, sensor_num):
    load_dir = os.path.join('output', data_dir)
    vis = Visualizer()
    #vis.error_plot(load_dir = load_dir)
    vis.load_data(load_dir = load_dir, data_type=data_type)
    vis.plot_3d(sensor_num= sensor_num)

if __name__ == '__main__':
    #run_model_kin_generator(model_name='seq_to_seq_3', participator=1, timesteps=1500, stride=10, nb_epoch=100, patience_limit=20, loss_delta_limit=0.1, load_weight_from = '2015-09-03 10:23:30.536226')
    run_visualizer('kin_2015-09-03 22:28:02.493086', data_type='train', sensor_num=2)
    #run_model_duration(model_name = 'lstm_3', participator = 1, timesteps = 100, stride = 1, nb_epoch = 10, load_weight_from = os.path.join('base_model', 'timestep_100_epoch_100'))
    #run_model_duration_generator(model_name = 'lstm_3', participator = 1, timesteps = 100, stride = 1, nb_epoch = 10, load_weight_from = os.path.join('base_model', 'timestep_100_epoch_100'))