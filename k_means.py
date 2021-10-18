# Bruno Iochins Grisci
# June 26th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

####################################################################################

if __name__ == '__main__': 
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
    out_fold = out_fold + 'kmeans/'

    if not os.path.exists(out_fold):
        os.makedirs(out_fold)

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    class_labels = list(np.sort(df[cfg.class_label].astype(str).unique()))
    feats_labels = list(df.columns.values)
    feats_labels.remove(cfg.class_label)

    if cfg.cv_splits is None:
        splits = RR_utils.split_cv(df, task=cfg.task, class_label=cfg.class_label, k=cfg.k)
        np.set_printoptions(threshold=sys.maxsize)
        with open(out_fold+"split.py", "w") as sf:
            sf.write("from collections import namedtuple\nfrom numpy import array\nSplit = namedtuple('Split', ['tr', 'te'])\nsplits = {}".format(splits))
        np.set_printoptions(threshold=1000)
    else:
        spt = importlib.import_module(cfg.cv_splits.replace('/','.').replace('.py',''))
        splits = spt.splits    

    ranks_values = []
    ranks_labels = []

    tr_accuracy = []
    te_accuracy = []

    for fold in range(len(splits)):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled)
        print(tr_df)

        X, Y = RR_utils.get_XY(tr_df, cfg.task, cfg.class_label)
        Y = tr_df[cfg.class_label].values
        
        clf = KMeans(n_clusters=2)
        clf = clf.fit(X)

        print(clf.cluster_centers_)
        
        y_pred = clf.predict(X)
        print(y_pred)
        if cfg.eval_metric == 'accuracy':
            tr_acc = metrics.accuracy_score(Y, y_pred)
            print("Train accuracy:", tr_acc)
        elif cfg.eval_metric == 'f1score':
            tr_acc = metrics.f1_score(Y, y_pred, average='macro')
            print("Train f1-score:", tr_acc)
        else:
            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))      

        tr_accuracy.append(tr_acc)
        print(metrics.confusion_matrix(Y, y_pred, labels=class_labels))

        if te_df.empty:
            l = [(tr_df, 'train')]
        else:
            l = [(tr_df, 'train'), (te_df, 'test')]
            teX, teY = RR_utils.get_XY(te_df, cfg.task, cfg.class_label)
            teY = te_df[cfg.class_label].values
            tey_pred = clf.predict(teX)
            if cfg.eval_metric == 'accuracy':
                te_acc = metrics.accuracy_score(teY, tey_pred)
                print("Test accuracy:", te_acc)
            elif cfg.eval_metric == 'f1score':
                te_acc = metrics.f1_score(teY, tey_pred, average='macro')
                print("Test f1-score:", te_acc)
            else:
                raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))               
            te_accuracy.append(te_acc)
            print(metrics.confusion_matrix(teY, tey_pred, labels=class_labels))

        ranks_labels.append(fold+1)

        for dataset in l:
            print('### {}:'.format(dataset[1]))

            C = dataset[0][cfg.class_label].nunique()
            D = len(dataset[0].columns) - 1
            N = len(df.index)
            print('\n{} classes, {} features, {} samples\n'.format(C, D, N)) 

    if cfg.eval_metric == 'accuracy':
        RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, '{}acc.txt'.format(out_fold))
    elif cfg.eval_metric == 'f1score':
        RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, '{}f1score.txt'.format(out_fold))
    else:
        raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))       
