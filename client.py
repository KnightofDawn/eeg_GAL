__author__ = 'jinwon'
import logging
import os
import datetime
import numpy as np
import time
from eeg_GAL import EEG_model, GAL_data
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

    rnn = EEG_model(event_list)
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

    event_list=['Dur_Reach', 'Dur_LoadReach', 'Dur_LoadMaintain', 'Dur_LoadRetract', 'Dur_Retract']

    rnn = EEG_model(event_list)
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

    rnn.save_event_generator_classify(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

def run_model_duration_generator_classify(model_name, participator, timesteps, stride, nb_epoch, load_weight_from = None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    output_dir = os.path.join('output', 'dur_classify_'+str(f_time))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info', 'kin'])
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    event_list=['Idle', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']
    rnn = EEG_model(event_list)
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
        generator = gal.data_generator_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)
        logger.info( 'epoch : {0}'.format(epoch))
        start = time.clock()
        rnn.run_model_with_generator_event_classify(generator=generator, train_list=train_list, validate_list=validate_list, test_list=test_list)
        logger.info( 'epoch {0} ran for {1} minutes'.format(epoch, (time.clock() - start)/60))


    rnn.set_data_description(data_description)
    rnn.set_model_config('epoch', nb_epoch)
    generator = gal.data_generator_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list)

    rnn.save_event_generator_classify(generator=generator,train_list=train_list, validate_list=validate_list,test_list=test_list, event_list=event_list, output_dir=output_dir)

def run_model_duration_classify(model_name, participator, timesteps, stride, max_freq, min_freq, nb_epoch, batch_size, save_by, multi_filter = False, load_weight_from = None, load_data_from= None):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(nb_epoch) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()
    if load_weight_from != None:
        load_weight_from_str = load_weight_from.split('/')
        output_dir = os.path.join('output', '{0}_P{1}_ts{2}_stride{3}_ep{4}_bs_{5}_weight_{6}_maxmin_{7}_saveby_{8}'.format(model_name, participator, timesteps, stride, nb_epoch, batch_size, load_weight_from_str, '{}_{}'.format(max_freq, min_freq), save_by))

    else:
        output_dir = os.path.join('output', '{0}_P{1}_ts{2}_stride{3}_ep{4}_bs_{5}_maxmin_{6}_saveby_{7}'.format(model_name, participator, timesteps, stride, nb_epoch, batch_size, '{}_{}'.format(max_freq, min_freq), save_by))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print('same configuation already exists!')
        return

    if load_data_from != None:
        assert timesteps == load_data_from[0], 'timesteps does not match'
        assert stride == load_data_from[1], 'stride does not match'


    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


    event_list=['Idle', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']

    rnn = EEG_model(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)
        logger.info('loaded weight')

    logger.info( 'running model data')

    data_split_ratio = [0.7, 0.2, 0.1]

    if load_data_from == None:
        gal = GAL_data()
        gal.set_logger(logger)
        gal.load_data(load_list=['eeg', 'info', 'kin'])
        if multi_filter ==True:
            gal.preprocess_filter_multiple()
        else:
            gal.preprocess_filter(max_freq=max_freq, min_freq=min_freq, low_pass=False)


        data_description = gal.get_data_description()
        participator = participator
        logger.info('participator : {0}'.format(participator))
        data = gal.data_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, partition_ratio=data_split_ratio, input_dim=32)
    else:
        data_description = dict()
        data_description['participator'] = participator
        data_description['timesteps'] = timesteps
        data_description['stride'] = stride
        data_description['event_list'] = event_list
        data_description['preprocess_filter'] = '{}_{}'.format(max_freq, min_freq)

        ts = load_data_from[0]
        st = load_data_from[1]
        data = list()
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'train_X_ts_{}_st_{}.npy'.format(ts, st))))
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'train_y_ts_{}_st_{}.npy'.format(ts, st))))
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'validation_X_ts_{}_st_{}.npy'.format(ts, st))))
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'validation_y_ts_{}_st_{}.npy'.format(ts, st))))
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'test_X_ts_{}_st_{}.npy'.format(ts, st))))
        data.append(np.load(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), 'test_y_ts_{}_st_{}.npy'.format(ts, st))))



    for i in range(nb_epoch/save_by):
        loss_train, loss_test = rnn.run_model_event(data=data, nb_epoch = save_by, batch_size=batch_size)
        output_dir_temp = os.path.join(output_dir, str(i))
        os.makedirs(output_dir_temp)
        df_loss = pd.DataFrame()
        df_loss['acc'] = loss_train.history['acc']
        df_loss['loss'] = loss_train.history['loss']
        df_loss.to_csv(os.path.join(output_dir_temp, 'train_loss_acc.csv'))




        rnn.set_data_description(data_description)
        rnn.set_model_config('epoch', nb_epoch/save_by * i)
        rnn.save_event_classify(data=data, event_list=event_list, output_dir=output_dir_temp)

def run_model_duration_classify_predict(model_name, participator, timesteps, stride, batch_size, save_by, load_weight_from):
    logger_name = model_name + str(participator) + str(timesteps) + str(stride) + str(load_weight_from)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()

    output_dir = os.path.join('output', 'predict_{0}_P{1}_ts{2}_stride{3}_bs_{4}_weight_{5}_{6}'.format(model_name, participator, timesteps, stride, batch_size,load_weight_from))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print('same configuation already exists!')
        return


    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info', 'kin'])
    gal.preprocess_filter(max_freq=50, min_freq=0.2, low_pass=False)
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    event_list=['Idle', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']
    rnn = EEG_model(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)

    rnn.load_model_weight(model_name, load_weight_from)
    logger.info('loaded weight')

    logger.info( 'running model data')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.1, 0.1]

    data = gal.data_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, partition_ratio=data_split_ratio, input_dim=32)
    rnn.set_data_description(data_description)
    rnn.save_event_classify(data=data, event_list=event_list, output_dir=output_dir)

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

    rnn = EEG_model(event_list)
    rnn.set_logger(logger)
    rnn.select_model(model_name)
    if load_weight_from != None:
        rnn.load_model_weight(model_name, load_weight_from)

    logger.info( 'running model data as a whole')
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

    rnn = EEG_model(None)
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

    rnn = EEG_model(None)
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

    rnn = EEG_model(None)
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

def get_data_multiple_filter(participator, timesteps, stride):
    logger_name = 'data' + str(participator) + str(timesteps) + str(stride)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()

    output_dir = os.path.join('output', '{0}_P{1}_ts{2}_stride{3}'.format('data', participator, timesteps, stride))


    os.makedirs(output_dir)

    hdlr = logging.FileHandler(os.path.join(output_dir, 'rnn.log'))
    logger.addHandler(hdlr)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info', 'kin'])
    gal.preprocess_filter_multiple()
    gal.save_individual_eeg()
    # data_split_ratio = [0.5, 0.3, 0.2]
    # event_list=['Idle', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']
    #
    #
    # data = gal.data_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, partition_ratio=data_split_ratio, input_dim=32 * 4)
    #
    # names = ['train_X', 'train_y', 'val_X', 'val_y', 'test_X', 'test_y']
    #
    # for i, j, in zip(data, names):
    #     np.save('{0}_{1}.npy'.format(j, stride), i)

def save_data_model_duration_classify(participator, timesteps, stride, max_freq, min_freq):
    logger_name = str(participator) + str(timesteps) + str(stride)
    logger = logging.getLogger(logger_name) # so that no multiple loggers input the same data
    f_time = datetime.datetime.today()

    gal = GAL_data()
    gal.set_logger(logger)
    gal.load_data(load_list=['eeg', 'info', 'kin'])
    gal.preprocess_filter(max_freq=max_freq, min_freq=min_freq, low_pass=False)

    # gal.preprocess_filter_multiple()
    data_description = gal.get_data_description()
    participator = participator
    logger.info('participator : {0}'.format(participator))

    event_list=['Idle', 'Reach_Phase', 'LoadReach_Phase', 'LoadMaintain_Phase', 'LoadRetract_Phase', 'Retract_Phase']


    logger.info( 'running model data')
    data_len=gal.part_data_count[participator]
    data_split_ratio = [0.8, 0.1, 0.1]

    data = gal.data_event_classify(part=participator, timesteps=timesteps, stride=stride, event_list=event_list, partition_ratio=data_split_ratio, input_dim=32)

    data_name = ['train', 'validation', 'test']

    if not os.path.exists(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq))):
        os.makedirs(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq)))

    for i in range(3):
        x = data[i * 2]
        y = data[i * 2 + 1]
        np.save(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), data_name[i]+'_X_ts_{}_st_{}.npy'.format(timesteps, stride)), x)
        np.save(os.path.join('data', 'numpy_binary', 'maxfreq_{}_minfreq_{}'.format(max_freq, min_freq), data_name[i]+'_y_ts_{}_st_{}.npy'.format(timesteps, stride)), y)

if __name__ == '__main__':
    #run_model_kin_generator(model_name='seq_to_seq_3', participator=1, timesteps=1500, stride=10, nb_epoch=100, patience_limit=20, loss_delta_limit=0.1, load_weight_from = '2015-09-03 10:23:30.536226')
    #run_visualizer('kin_2015-09-03 22:28:02.493086', data_type='train', sensor_num=2)
    #run_model_duration(model_name = 'simple_rnn_1', participator = 1, timesteps = 100, stride = 1, nb_epoch = 10)

    # for part in np.arange(12) + 1:
    #     run_model_duration_generator(model_name = 'lstm_3', participator = part, timesteps = 10, stride = 1, nb_epoch = 100, load_weight_from=os.path.join('base_model', 'timestep_10_epoch_100'))

    # run_model_duration_generator(model_name = 'lstm_5', participator = 1, timesteps = 100, stride = 1, nb_epoch = 200, load_weight_from='dur_2015-09-13 20:35:39.830666')

    # run_model_duration_generator(model_name = 'lstm_3_softmax', participator = 1, timesteps = 50, stride = 1, nb_epoch = 50)

    #run_model_duration_generator(model_name = 'lstm_3', participator = 12, timesteps = 10, stride = 1, nb_epoch = 80, load_weight_from=os.path.join('base_model', 'timestep_10_epoch_100'))
    #run_model_duration(model_name = 'lstm_3', participator = 1, timesteps = 10, stride = 1, nb_epoch = 10)

    '''
       now
    '''

    # run_model_duration_classify(model_name = 'gru_softmax', participator = 1, timesteps = 11, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'gru_softmax', participator = 1, timesteps = 51, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'gru_softmax', participator = 1, timesteps = 101, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'gru_softmax', participator = 1, timesteps = 201, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'lstm_softmax', participator = 1, timesteps = 12, stride = 100, nb_epoch = 20, batch_size = 1024)
    # run_model_duration_classify(model_name = 'lstm_softmax', participator = 1, timesteps = 51, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'lstm_softmax', participator = 1, timesteps = 101, stride = 10, nb_epoch = 100, batch_size = 128)
    # run_model_duration_classify(model_name = 'lstm_softmax', participator = 1, timesteps = 201, stride = 10, nb_epoch = 100, batch_size = 128)



    # run_model_duration_classify_predict(model_name = 'lstm_softmax_5', participator = 1, timesteps = 256, stride = 10, batch_size = 512, save_by=5, load_weight_from='sample')

    # run_model_duration_classify(model_name = 'gru_softmax_5', participator = 1, timesteps = 256, nb_epoch=100, stride = 10, batch_size = 512, save_by=20)
    # run_model_duration_classify(model_name = 'jzs1_softmax_5', participator = 1, timesteps = 256, nb_epoch=80, stride = 10, batch_size = 512, save_by=20)
    # run_model_duration_classify(model_name = 'jzs2_softmax_5', participator = 1, timesteps = 256, nb_epoch=80, stride = 10, batch_size = 512, save_by=20)
    # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 256, nb_epoch=80, stride = 10, batch_size = 512, save_by=20)


    # get_data_multiple_filter(participator=1, timesteps=256, stride=1)
    # get_data_multiple_filter(participator=1, timesteps=128, stride=1)
    # get_data_multiple_filter(participator=1, timesteps=16, stride=1)

    ###########################################################################################################################
    '''
    for saving data
    '''


    # for min_freq, max_freq in zip([0.5, 4, 8, 13], [4, 8, 13, 30]):
        # save_data_model_duration_classify(participator=1, timesteps=1, stride=10, max_freq=max_freq, min_freq=min_freq)
        # save_data_model_duration_classify(participator=1, timesteps=10, stride=10, max_freq=max_freq, min_freq=min_freq)
        # save_data_model_duration_classify(participator=1, timesteps=100, stride=10, max_freq=max_freq, min_freq=min_freq)
        # save_data_model_duration_classify(participator=1, timesteps=200, stride=10, max_freq=max_freq, min_freq=min_freq)
        # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 1, stride = 10, max_freq = max_freq, min_freq = min_freq, nb_epoch=201, batch_size = 512, load_data_from=[1,10], save_by=50)
        # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 10, stride = 10, max_freq = max_freq, min_freq = min_freq, nb_epoch=201, batch_size = 512, load_data_from=[10,10], save_by=50)
        # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 200, stride = 10, max_freq = max_freq, min_freq = min_freq, nb_epoch=101, batch_size = 512, load_data_from=[200,10], save_by=50)

    # for min_freq, max_freq in zip([0.1, 0.1], [30, 1]):
    #     run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 200, stride = 10, max_freq = max_freq, min_freq = min_freq, nb_epoch=201, batch_size = 256, load_data_from=[200,10], save_by=50)

    # save_data_model_duration_classify(participator=1, timesteps=1, stride=1, max_freq=1, min_freq=0.1)
    # save_data_model_duration_classify(participator=1, timesteps=10, stride=1)

    # save_data_model_duration_classify(participator=1, timesteps=10, stride=10, max_freq=128, min_freq=128)
    # save_data_model_duration_classify(participator=1, timesteps=10, stride=10, max_freq=256, min_freq=256)


    run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 200, stride = 10, max_freq = 30, min_freq = 0.1, nb_epoch=3, batch_size = 256, save_by=200, load_weight_from='jzs3_softmax_5_P1_ts200_stride10_ep501_bs_256_maxmin_30_0.1_saveby_50/0')
    # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 100, stride = 10, max_freq = 30, min_freq = 0.1, nb_epoch=501, batch_size = 512, load_data_from=[100,10], save_by=50)
    # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 200, stride = 10, max_freq = 30, min_freq = 0.1, nb_epoch=501, batch_size = 512, load_data_from=[200,10], save_by=50)

    # run_model_duration_classify(model_name = 'jzs3_softmax_5', participator = 1, timesteps = 10, stride = 10, max_freq = 256, min_freq = 256, nb_epoch=501, batch_size = 128, load_data_from=[10,10], save_by=50)

