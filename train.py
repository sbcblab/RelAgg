# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.utils import class_weight

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

tensorflow.compat.v1.enable_eager_execution() 

####################################################################################       

if __name__ == '__main__': 
    print(tensorflow.__version__)
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    if not os.path.exists(out_fold+'network_eval/'):
        os.makedirs(out_fold+'network_eval/')

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    if cfg.task == 'regression':
        C = 1
        df[cfg.class_label] = RR_utils.shift_target(df, cfg.class_label)
        df[cfg.class_label] = df[cfg.class_label]/df[cfg.class_label].max()
    elif cfg.task == 'classification':
        C =  df[cfg.class_label].nunique()
    else:
        raise Exception('Unknown task type: {}'.format(cfg.task))
    D = len(df.columns) - 1
    N = len(df.index)

    print("Original dataset: {}\n".format(cfg.dataset_file))
    if cfg.task == 'classification':
        print('\nClasses:')
        print(np.sort(df[cfg.class_label].unique()))
    elif cfg.task == 'regression':
        print('\nTarget:')
        print('{} +- {}, [{}, {}]'.format(round(df[cfg.class_label].mean(), 2), round(df[cfg.class_label].std(), 2), df[cfg.class_label].min(), df[cfg.class_label].max()))
    print(df)
    print('\n{} classes, {} features, {} samples\n'.format(C, D, N)) 

    if cfg.batch_size > N:
        raise Exception('Batch size ({}) is larger than the number of samples ({})!'.format(cfg.batch_size, N))
    elif cfg.batch_size == N:
        print('\nWARNING: batch size ({}) is equal to the number of samples ({})!\n'.format(cfg.batch_size, N))

    cp_list = list(range(cfg.checkpoint, cfg.train_epochs+1, cfg.checkpoint))
    if cp_list[-1] != cfg.train_epochs:
        cp_list.append(cfg.train_epochs)

    ###################################################################################

    if cfg.cv_splits is None:
        splits = RR_utils.split_cv(df, task=cfg.task, class_label=cfg.class_label, k=cfg.k)
        np.set_printoptions(threshold=sys.maxsize)
        with open(out_fold+"split.py", "w") as sf:
            sf.write("from collections import namedtuple\nfrom numpy import array\nSplit = namedtuple('Split', ['tr', 'te'])\nsplits = {}".format(splits))
        np.set_printoptions(threshold=1000)
    else:
        #spt = importlib.import_module(cfg.cv_splits.replace('/','.').replace('.py',''))
        spec = importlib.util.spec_from_file_location(cfg.cv_splits.replace('/','.').replace('.py',''), cfg.cv_splits) 
        spt  = importlib.util.module_from_spec(spec) 
        spec.loader.exec_module(spt)         
        splits = spt.splits

    tr_accuracy = []
    te_accuracy = []

    for fold in range(len(splits)):

        print('###### CROSS-VALIDATION {}-FOLD:'.format(fold+1))

        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled) 
        X, Y = RR_utils.get_XY(tr_df, cfg.task, cfg.class_label)
        teX, teY = RR_utils.get_XY(te_df, cfg.task, cfg.class_label)

        if cfg.batch_size > len(X):
            raise Exception('Batch size ({}) is larger than the number of samples of training set ({}) in fold {}!'.format(cfg.batch_size, len(X), fold+1))
        elif cfg.batch_size == len(X):
            print('\nWARNING: Batch size ({}) is equal to the number of samples of training set ({}) in fold {}!\n'.format(cfg.batch_size, len(X), fold+1))

        if teX is not None:
            if cfg.batch_size > len(teX):
                raise Exception('Batch size ({}) is larger than the number of samples of test set ({}) in fold {}!'.format(cfg.batch_size, len(teX), fold+1))
            elif cfg.batch_size == len(teX):
                print('\nWARNING: Batch size ({}) is equal to the number of samples of test set ({}) in fold {}!\n'.format(cfg.batch_size, len(teX), fold+1))

        if cfg.task == 'classification':
            class_weights_index = RR_utils.get_class_weights(np.sort(df[cfg.class_label].unique()), tr_df[cfg.class_label].values, index_type='index')
            print('\nClass weights:\n', class_weights_index, '\n')


        regularization = cfg.regularization
        
        if cfg.regularizer == 'l1':
            l_reg = l1
        elif cfg.regularizer == 'l2':
            l_reg = l2
        elif cfg.regularizer is None:
            l_reg = l1
            regularization = 0.0
        else:
            raise Exception("Regularizer not found: {}".format(cfg.regularizer))

        K.clear_session()
        input_layer_exists = False
        model = Sequential()
        for layer in cfg.layers:
            if layer[0].lower().strip() == 'dense':
                if input_layer_exists:
                    if cfg.weights_constraints:
                        model.add(Dense(int(layer[1]), activation=layer[2].lower().strip(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization), bias_constraint=RR_utils.NonPos()))
                    else:
                        model.add(Dense(int(layer[1]), activation=layer[2].lower().strip(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization)))
                else:
                    if cfg.weights_constraints:
                        model.add(Dense(int(layer[1]), activation=layer[2].lower().strip(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization), bias_constraint=RR_utils.NonPos(), input_dim=D))
                    else:
                        model.add(Dense(int(layer[1]), activation=layer[2].lower().strip(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization), input_dim=D))
                    input_layer_exists = True
            elif layer[0].lower().strip() == 'dropout':
                model.add(Dropout(float(layer[1])))
            else:
                raise Exception('Unknown layer type: {} in {}'.format(layer[0], layer))
        

        if cfg.task == 'classification':
            actfun = 'softmax'
        elif cfg.task == 'regression':
            actfun = 'linear'

        if cfg.weights_constraints:
            model.add(Dense(C, activation=actfun, kernel_constraint=non_neg(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization), bias_constraint=RR_utils.IsZero()))
        else:
            model.add(Dense(C, activation=actfun, kernel_constraint=non_neg(), kernel_regularizer=l_reg(regularization), bias_regularizer=l_reg(regularization), bias_constraint=non_neg()))
            #model.add(Dense(C, activation=actfun))

        #optimizer = 'adam'
        optimizer = 'SGD'
        checkpoint = ModelCheckpoint("{}_{}_".format(out_file, fold+1)+"{epoch:04d}.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='auto', period=cfg.checkpoint)
        csv_logger = CSVLogger(out_fold + 'network_eval/{}_log.csv'.format(fold+1), append=True, separator=',')
        if cfg.task == 'classification':
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], weighted_metrics=['accuracy'])
            # Fit the model
            model.fit(X, Y, epochs=cfg.train_epochs, batch_size=cfg.batch_size, shuffle=True, class_weight=class_weights_index, verbose=2, callbacks=[checkpoint, csv_logger])
        elif cfg.task == 'regression':
            # Compile model
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            # Fit the model
            model.fit(X, Y, epochs=cfg.train_epochs, batch_size=cfg.batch_size, shuffle=True, verbose=2, callbacks=[checkpoint, csv_logger])

        #save the network
        model.save("{}_{}_{:04d}.hdf5".format(out_file, fold+1, cfg.train_epochs), include_optimizer = False)
        plot_model(model, show_shapes=True, show_layer_names=True, to_file="{}{}/{}.png".format(out_fold, 'network_eval', fold+1))

        
        # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
        y_pred1 = model.predict(X)

        if cfg.task == 'classification':
            y_pred  = np.argmax(y_pred1, axis=1)
            y       = np.argmax(Y, axis=1)

            if cfg.eval_metric == 'accuracy':
                tr_accuracy.append(metrics.accuracy_score(y, y_pred))
                print('\nTraining accuracy: {}'.format(tr_accuracy[-1]))
            elif cfg.eval_metric == 'f1score':
                tr_accuracy.append(metrics.f1_score(y, y_pred, average="macro"))
                print('\nTraining F1-score: {}'.format(tr_accuracy[-1]))
            else:
                raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
        elif cfg.task == 'regression':
            tr_accuracy.append(metrics.mean_squared_error(Y, y_pred1))
            print('\nTraining MSE: {}'.format(tr_accuracy[-1]))

        if teX is not None:
            tey_pred1 = model.predict(teX)
            if cfg.task == 'classification':
                tey_pred  = np.argmax(tey_pred1, axis=1)
                tey       = np.argmax(teY, axis=1)
                if cfg.eval_metric == 'accuracy':
                    te_accuracy.append(metrics.accuracy_score(tey, tey_pred))
                    print('\nTest accuracy: {}\n'.format(te_accuracy[-1]))
                elif cfg.eval_metric == 'f1score':
                    te_accuracy.append(metrics.f1_score(tey, tey_pred, average="macro"))
                    print('\nTest F1-score: {}\n'.format(te_accuracy[-1]))
                else:
                    raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
            elif cfg.task == 'regression':
                te_accuracy.append(metrics.mean_squared_error(teY, tey_pred1))
                print('\nTest MSE: {}\n'.format(te_accuracy[-1])) 
        else:
            te_accuracy.append(None)

        if cfg.task == 'classification':
            RR_utils.confusion_matrix(tr_df, cfg.class_label, model, out_fold+"network_eval/{}_TR_cm.csv".format(fold+1))
            if teX is not None:
                RR_utils.confusion_matrix(te_df, cfg.class_label, model, out_fold+"network_eval/{}_TE_cm.csv".format(fold+1))
        elif cfg.task == 'regression':
            RR_utils.regression_distance(tr_df, cfg.class_label, model, out_fold+"network_eval/{}_TR_regdist.csv".format(fold+1))
            if teX is not None:
                RR_utils.regression_distance(te_df, cfg.class_label, model, out_fold+"network_eval/{}_TE_regdist.csv".format(fold+1))
   
        for cp in cp_list: 
            model_cp = load_model('{}_{}_{:04d}.hdf5'.format(out_file, fold+1, cp), custom_objects={'NonPos': RR_utils.NonPos, 'IsZero': RR_utils.IsZero}, compile=False)        
            RR_utils.save_outputs(df, splits[fold], cfg.task, cfg.class_label, model_cp, out_file+"_{}_{:04d}_out.csv".format(fold+1, cp), rescale=cfg.rescaled, standard=cfg.standardized, minVals=min_vals, maxVals=max_vals, meanVals=mean_vals, stdVals=std_vals)
 
    ###################################################################################
    
    if cfg.task == 'classification':
        if cfg.eval_metric == 'accuracy':
            RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, out_fold +'network_eval/acc.txt')
        elif cfg.eval_metric == 'f1score':
            RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, out_fold +'network_eval/f1score.txt')
        else:
            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
    elif cfg.task == 'regression':
        RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, out_fold +'network_eval/mse.txt')