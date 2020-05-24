# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

SCORE_LABEL = 'score'

####################################################################################

def get_selection_txt(file_name):
    with open(file_name, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines

def get_selection(file_name, label, CLASS_LABELS):
    df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    if cfg.task == 'regression':
        label = SCORE_LABEL
    if label == 'combine':
        all_ids = []
        i = 0
        #while (len(all_ids) < max(cfg.n_selection, 2*len(CLASS_LABELS))) and i < cfg.n_selection:
        for cl in CLASS_LABELS:
            all_ids.append(df[cl].values[0])
            all_ids = list(set(all_ids))
        i = i+1
        all_ids = list(set(all_ids))       
        return list(set(df.values.flatten()))
    else:
        return list(df[label].values)

def multiple_runs():
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
    if cfg.task == 'classification':
        class_labels = list(np.sort(pd.read_csv(cfg.dataset_file, delimiter=',', header=0, index_col=0)[cfg.class_label].unique()))
        class_labels = [str(cl) for cl in class_labels]    
    elif cfg.task == 'regression':
        class_labels = ['target']

    list_acc_tr = {}
    list_acc_te = {}
    for file_label in class_labels + [SCORE_LABEL, 'combine', 'mrmr', 'kw', 'tree', 'original']:
        list_acc_tr[file_label] = []
        list_acc_te[file_label] = []        

    for fold in range(1, cfg.k+1):
        for use in ['train']:
            print('\n', fold, use, '\n')
            file_names = []

            for file_label in class_labels + [SCORE_LABEL, 'combine', 'mrmr', 'kw', 'tree', 'original']:
                if file_label == 'mrmr':
                    file = ('{}mrmr/{}_{}_selection.txt'.format(out_fold, fold, use), file_label)
                elif file_label == 'kw':
                    file = ('{}kw/{}_{}_selection.txt'.format(out_fold, fold, use), file_label)
                elif file_label == 'tree':
                    file = ('{}tree/{}_features.txt'.format(out_fold, fold), file_label)
                elif file_label == 'original':
                    file = ('all', file_label)
                elif file_label == 'ww':
                    file = ('{}_{}_{}_selection.csv'.format(out_file.replace(out_fold, out_fold+'ww/'), fold, use), file_label)
                else:
                    file = ('{}/relevance_eval/{}_{}_selection.csv'.format(out_fold, fold, use), file_label)

                try:
                    if file[1] == 'mrmr':
                        selection = get_selection_txt(file[0])
                    elif file[1] == 'kw':
                        selection = get_selection_txt(file[0])
                    elif file[1] == 'tree':
                        selection = get_selection_txt(file[0])
                    elif file[1] == 'original':
                        selection = 'all'
                    elif file[1] == 'ww':
                        selection = get_selection(file[0], SCORE_LABEL, class_labels)
                    else:
                        selection = get_selection(file[0], file[1], class_labels)

                    tr_acc, te_acc = run(selection, '{}_{}_{}.txt'.format(use, fold, file[1]))
                    list_acc_tr[file_label].append(tr_acc)
                    list_acc_te[file_label].append(te_acc)

                    print(file)
                    file_names.append(file)

                except:
                    print('Could not open file {}'.format(file))

    save_tr = {}
    save_te = {} 
    indexes = class_labels + [SCORE_LABEL, 'combine', 'mrmr', 'kw', 'tree', 'original']
    save_tr[cfg.eval_metric] = []
    save_te[cfg.eval_metric] = []
    save_tr['std'] = []
    save_te['std'] = []     
    for file_label in indexes:
        save_tr[cfg.eval_metric].append(np.array(list_acc_tr[file_label]).mean())
        save_te[cfg.eval_metric].append(np.array(list_acc_te[file_label]).mean())
        save_tr['std'].append(np.array(list_acc_tr[file_label]).std())
        save_te['std'].append(np.array(list_acc_te[file_label]).std())
    acc_tr_df = pd.DataFrame(save_tr, index=indexes)  
    acc_te_df = pd.DataFrame(save_te, index=indexes)
    acc_tr_df.to_csv(out_fold +'svm/'+'training.csv')
    acc_te_df.to_csv(out_fold +'svm/'+'testing.csv')
    print(acc_tr_df)
    print(acc_te_df)

def run(features, save_file):

    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    if not os.path.exists(out_fold + 'svm/'):
        os.makedirs(out_fold + 'svm/')

    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))
    splits = spt.splits

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    if cfg.task == 'classification':
        class_labels = list(np.sort(df[cfg.class_label].astype(str).unique()))
    elif cfg.task == 'regression':
        class_labels = ['target']
    
    if features != 'all':
        print('Selected {} features: {}\n'.format(len(features), features))
        features.append(cfg.class_label)
        df = df[features]

    feats_labels = list(df.columns.values)
    feats_labels.remove(cfg.class_label)

    ranks_values = []
    ranks_labels = []

    tr_accuracy = []
    te_accuracy = []

    for fold in range(len(splits)):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        #out = pd.read_csv('{}_{}_out.csv'.format(out_file, fold+1), delimiter=cfg.dataset_sep, header=0, index_col=0)
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled)
        print(tr_df)

        X, Y = RR_utils.get_XY(tr_df, cfg.task, cfg.class_label)
        Y = tr_df[cfg.class_label].values

        if cfg.task == 'classification':
            clf = svm.SVC(class_weight='balanced')
        elif cfg.task == 'regression':
            clf = svm.SVR()
        clf = clf.fit(X, Y)
        
        y_pred = clf.predict(X)
        if cfg.task == 'classification':
            if cfg.eval_metric == 'accuracy':
                tr_acc = metrics.accuracy_score(Y, y_pred)
                print("Train accuracy:", tr_acc)
            elif cfg.eval_metric == 'f1score':
                tr_acc = metrics.f1_score(Y, y_pred, average='macro')
                print("Train f1-score:", tr_acc)
            else:
                raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))            
        elif cfg.task == 'regression':
            tr_acc = metrics.mean_squared_error(Y, y_pred)
            print("Train MSE:", tr_acc) 
        tr_accuracy.append(tr_acc)
        if cfg.task == 'classification':
            print(metrics.confusion_matrix(Y, y_pred, labels=class_labels))

        if te_df.empty:
            l = [(tr_df, 'train')]
        else:
            l = [(tr_df, 'train'), (te_df, 'test')]
            teX, teY = RR_utils.get_XY(te_df, cfg.task, cfg.class_label)
            teY = te_df[cfg.class_label].values
            tey_pred = clf.predict(teX)
            if cfg.task == 'classification':
                if cfg.eval_metric == 'accuracy':
                    te_acc = metrics.accuracy_score(teY, tey_pred)
                    print("Test accuracy:", te_acc)
                elif cfg.eval_metric == 'f1score':
                    te_acc = metrics.f1_score(teY, tey_pred, average='macro')
                    print("Test f1-score:", te_acc)
                else:
                    raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))
            elif cfg.task == 'regression':
                te_acc = metrics.mean_squared_error(teY, tey_pred)
                print("Test MSE:", te_acc) 
            te_accuracy.append(te_acc)
            if cfg.task == 'classification':
                print(metrics.confusion_matrix(teY, tey_pred, labels=class_labels))

    RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, out_fold +'svm/'+save_file)
    return np.array(tr_accuracy).mean(), np.array(te_accuracy).mean()

if __name__ == '__main__':
    multiple_runs()
