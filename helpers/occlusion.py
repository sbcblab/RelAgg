# Bruno Iochins Grisci
# May 24th, 2020

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np ; na = np.newaxis
import pandas as pd
import importlib
from collections import namedtuple
from tensorflow.keras.models import load_model
from sklearn import metrics
from copy import deepcopy

import model_io
import RR_utils

config_file     = sys.argv[1]
reference_class = int(sys.argv[2])
only_classes    = int(sys.argv[3])
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

SCORE_LABEL = 'score'

colhex = {
    'RED':     '#BA0000',
    'BLUE':    '#0000FF',
    'YELLOW':  '#FFEE00',
    'GREEN':   '#048200',    
    'ORANGE':  '#FF6103',
    'BLACK':   '#000000',
    'CYAN':    '#00FFD4',    
    'SILVER':  '#c0c0c0',
    'MAGENTA': '#680082',
    'CREAM':   '#FFFDD0',
    'DRKBRW':  '#654321',
    'BEIGE':   '#C2C237',
    'WHITE':   '#FFFFFF',
}

STEP = 1

##########################################################################################

if __name__ == '__main__':

    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:    
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))
    splits = spt.splits

    if not os.path.exists(out_fold+'relevance_eval/'):
        os.makedirs(out_fold+'relevance_eval/')
    if not os.path.exists(out_fold+'selection_eval/'):
        os.makedirs(out_fold+'selection_eval/')

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    if cfg.task == 'classification':
        CLASSES = np.sort(df[cfg.class_label].unique())
    elif cfg.task == 'regression':
        CLASSES = np.array(['target'])

    fold_accuracy_regular  = []
    fold_accuracy_reversed = []
    fold_accuracy_random   = []
    #fold_accuracy_ww       = []
    fold_accuracy_classes  = []
    for fold in range(len(splits)):
        
        print('\n###### {}-FOLD:\n'.format(fold+1))
        out = pd.read_csv('{}_{}_out.csv'.format(out_file, fold+1), delimiter=cfg.dataset_sep, header=0, index_col=0)
        #load a neural network
        print('# Reading neural network')
        model = load_model('{}_{}.hdf5'.format(out_file, fold+1), custom_objects={'NonPos': RR_utils.NonPos, 'IsZero': RR_utils.IsZero}, compile=False)       

        data_file = '{}_{}_{}_datasheet.csv'.format(out_file, fold+1, 'test')
        dat = pd.read_csv(data_file, header=0, index_col=0, low_memory=False)

        scores = dat[SCORE_LABEL].copy()
        scores.dropna(inplace=True)

        if cfg.task == 'classification':
            class_scores = {}
            for c in CLASSES:
                class_scores[c] = dat.sort_values([c], ascending=False)[c].copy().dropna()

        reversed_scores = scores.iloc[::-1]
        random_scores   = scores.sample(frac=1)

        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled)  

        if cfg.task == 'classification':
            classes_te = {}
            classes_ac = {}
            for c in CLASSES:
                classes_te[c] = te_df[te_df[cfg.class_label] == c]
                classes_ac[c] = []

        if only_classes == 0:
            if cfg.task == 'classification':
                print('= classes')
                ciclone = deepcopy(classes_te)

                fi = 0
                for feat in ['oOoOo'] + list(class_scores[CLASSES[reference_class]].index):
                    if feat != 'oOoOo':
                        for c in CLASSES:
                            #ciclone[c][feat] = -ciclone[c][feat]
                            ciclone[c][feat] = 0.0
                    if fi%STEP==0:
                        for c in CLASSES:
                            c_teX, sdnf = RR_utils.get_XY(ciclone[c], cfg.task, cfg.class_label)
                            c_teY    = np.tile((np.array(CLASSES) == c)*1, sdnf.shape)
                            c_tey_pred1  = model.predict(c_teX)
                            #print(c_tey_pred1)
                            ctp = []
                            for l in c_tey_pred1:
                                fm = np.where(l == np.max(l))[0]
                                if len(fm) == 1 or fm[0] != list(CLASSES).index(c):
                                    ctp.append(fm[0])
                                elif len(fm) > 1 and fm[0] == list(CLASSES).index(c):
                                    ctp.append(fm[1])
                            c_tey_pred  = np.array(ctp)                   
                            c_tey       = np.argmax(c_teY, axis=1)
                            classes_ac[c].append(metrics.accuracy_score(c_tey, c_tey_pred))  
                        print('=== {}: {}'.format(fi, classes_ac[CLASSES[reference_class]][-1]))
                        #1/0
                    fi = fi+1  
                fold_accuracy_classes.append(classes_ac)       

        if only_classes == 1:
            print('= score')
            fi = 0        
            clone = te_df.copy()
            feat_accuracy_regular = []
            for feat in ['oOoOo'] + list(scores.index):
                if feat != 'oOoOo':
                    clone[feat] = 0.0
                    #clone[feat] = -clone[feat]
                if fi%STEP==0:
                    teX, teY = RR_utils.get_XY(clone, cfg.task, cfg.class_label)
                    tey_pred1 = model.predict(teX)
                    if cfg.task == 'classification':
                        tey_pred  = np.argmax(tey_pred1, axis=1)
                        tey       = np.argmax(teY, axis=1)
                        if cfg.eval_metric == 'accuracy':
                            feat_accuracy_regular.append(metrics.accuracy_score(tey, tey_pred))
                        elif cfg.eval_metric == 'f1score':
                            feat_accuracy_regular.append(metrics.f1_score(tey, tey_pred, average="macro"))
                        else:
                            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))               

                    elif cfg.task == 'regression':
                        feat_accuracy_regular.append(metrics.mean_squared_error(teY, tey_pred1))

                    print('=== {}: {}'.format(fi, feat_accuracy_regular[-1]))
                fi = fi+1    
            fold_accuracy_regular.append(feat_accuracy_regular)

            print('= reversed')
            fi = 0  
            clone = te_df.copy()
            feat_accuracy_reversed = []
            for feat in ['oOoOo'] + list(reversed_scores.index):
                if feat != 'oOoOo':
                    clone[feat] = 0.0
                    #clone[feat] = -clone[feat]
                if fi%STEP==0:
                    teX, teY = RR_utils.get_XY(clone, cfg.task, cfg.class_label)
                    tey_pred1 = model.predict(teX)
                    if cfg.task == 'classification':
                        tey_pred  = np.argmax(tey_pred1, axis=1)
                        tey       = np.argmax(teY, axis=1)
                        if cfg.eval_metric == 'accuracy':
                            feat_accuracy_reversed.append(metrics.accuracy_score(tey, tey_pred))
                        elif cfg.eval_metric == 'f1score':
                            feat_accuracy_reversed.append(metrics.f1_score(tey, tey_pred, average="macro"))
                        else:
                            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
                    elif cfg.task == 'regression':
                        feat_accuracy_reversed.append(metrics.mean_squared_error(teY, tey_pred1))

                    print('=== {}: {}'.format(fi, feat_accuracy_reversed[-1]))
                fi = fi+1                    
            fold_accuracy_reversed.append(feat_accuracy_reversed)

            print('= random')
            fi = 0  
            clone = te_df.copy()
            feat_accuracy_random = []
            for feat in ['oOoOo'] + list(random_scores.index):
                if feat != 'oOoOo':
                    clone[feat] = 0.0
                    #clone[feat] = -clone[feat]
                if fi%STEP==0:
                    teX, teY = RR_utils.get_XY(clone, cfg.task, cfg.class_label)
                    tey_pred1 = model.predict(teX)
                    if cfg.task == 'classification':
                        tey_pred  = np.argmax(tey_pred1, axis=1)
                        tey       = np.argmax(teY, axis=1)
                        if cfg.eval_metric == 'accuracy':
                            feat_accuracy_random.append(metrics.accuracy_score(tey, tey_pred))
                        elif cfg.eval_metric == 'f1score':
                            feat_accuracy_random.append(metrics.f1_score(tey, tey_pred, average="macro"))
                        else:
                            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
                    elif cfg.task == 'regression':
                        feat_accuracy_random.append(metrics.mean_squared_error(teY, tey_pred1))

                    print('=== {}: {}'.format(fi, feat_accuracy_random[-1]))
                fi = fi+1                    
            fold_accuracy_random.append(feat_accuracy_random)           
        
    if only_classes == 1:
        decay = pd.Series(fold_accuracy_regular)
        print(decay)

        decay = pd.Series(fold_accuracy_reversed)
        print(decay)

        decay = pd.Series(fold_accuracy_random)
        print(decay)

        fold_accuracy_regular = np.array(fold_accuracy_regular)
        mean_accuracy_regular = fold_accuracy_regular.mean(axis=0)
        std_accuracy_regular  = fold_accuracy_regular.std(axis=0)

        fold_accuracy_reversed = np.array(fold_accuracy_reversed)
        mean_accuracy_reversed = fold_accuracy_reversed.mean(axis=0)
        std_accuracy_reversed  = fold_accuracy_reversed.std(axis=0)

        fold_accuracy_random = np.array(fold_accuracy_random)
        mean_accuracy_random = fold_accuracy_random.mean(axis=0)
        std_accuracy_random  = fold_accuracy_random.std(axis=0)       
        
        decay = pd.DataFrame([])
        decay['score']       = mean_accuracy_regular
        decay['_std1_score'] = mean_accuracy_regular + std_accuracy_regular
        decay['_std2_score'] = mean_accuracy_regular - std_accuracy_regular

        decay['reversed']       = mean_accuracy_reversed
        decay['_std1_reversed'] = mean_accuracy_reversed + std_accuracy_reversed
        decay['_std2_reversed'] = mean_accuracy_reversed - std_accuracy_reversed

        decay['random']       = mean_accuracy_random
        decay['_std1_random'] = mean_accuracy_random + std_accuracy_random
        decay['_std2_random'] = mean_accuracy_random - std_accuracy_random  

        if cfg.task == 'classification':
            decay = decay.clip(upper=1.0)
        decay = decay.clip(lower=0.0)

        print(decay)
        decay.to_csv(data_file.replace('.csv', '_occ.csv'))
        decay.plot.line(style=['b-', 'b:', 'b:', 'r-', 'r:', 'r:', 'k-', 'k:', 'k:']).get_figure().savefig(data_file.replace('.csv', '_occ.pdf'))        
        plt.clf()

    if only_classes == 0:
        if cfg.task == 'classification':

            classes_decay = pd.DataFrame([])
            for c in CLASSES: 
                fdf = pd.DataFrame([])
                for fa in range(len(fold_accuracy_classes)):
                    fdf[fa] = fold_accuracy_classes[fa][c]
                classes_decay[c] = fdf.mean(axis=1)
                #classes_decay['_std1'+c] = fdf.mean(axis=1) + fdf.std(axis=1) 
                #classes_decay['_std2'+c] = fdf.mean(axis=1) - fdf.std(axis=1)

            classes_decay = classes_decay.clip(upper=1.0)
            classes_decay = classes_decay.clip(lower=0.0)
            classes_decay.set_index(np.array(range(0, len(classes_decay[CLASSES[0]])))*STEP, inplace=True)
            classes_decay.to_csv(data_file.replace('.csv', '_{}_occ.csv'.format(CLASSES[reference_class])))

            fig = None
            for cn in range(len(CLASSES)):
                if cn == reference_class:
                    fig = classes_decay[CLASSES[cn]].plot.line(label=CLASSES[cn], legend=True, color=colhex[cfg.class_colors[cn]], linestyle='-',  marker='s')
                else:
                    fig = classes_decay[CLASSES[cn]].plot.line(label=CLASSES[cn], legend=True, color=colhex[cfg.class_colors[cn]], linestyle='-')
                #fig = classes_decay['_std1'+CLASSES[cn]].plot.line(color=colhex[cfg.class_colors[cn]], linestyle=':')
                #fig = classes_decay['_std2'+CLASSES[cn]].plot.line(color=colhex[cfg.class_colors[cn]], linestyle=':')
            
            #print(np.array(range(0, len(classes_decay[CLASSES[0]]))))
            #fig.set_xticklabels(np.array(range(0, len(classes_decay[CLASSES[0]])))*STEP)
            fig.get_figure().savefig(data_file.replace('.csv', '_{}_occ.pdf'.format(CLASSES[reference_class])))
