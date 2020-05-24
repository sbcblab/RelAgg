# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.tree.export import export_text
import graphviz

import RR_utils
import plot_pca

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

####################################################################################

if __name__ == '__main__': 
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    if not os.path.exists(out_fold + 'tree/'):
        os.makedirs(out_fold + 'tree/')

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
        class_labels = ['Target']
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
            clf = tree.DecisionTreeClassifier(min_samples_leaf=0.1, class_weight='balanced')
        elif cfg.task == 'regression':
            clf = tree.DecisionTreeRegressor(min_samples_leaf=0.1)
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

        dot_data = tree.export_graphviz(clf,out_file=None, 
                                            feature_names=feats_labels,  
                                            class_names=class_labels,  
                                            filled=True, rounded=True,  
                                            special_characters=True)
        graph = graphviz.Source(dot_data) 
        graph.render('{}tree/{}'.format(out_fold, fold+1)) 

        r = export_text(clf, feature_names=feats_labels)
        print(r)
        with open('{}tree/{}_rules.txt'.format(out_fold, fold+1), 'w') as tree_rule:
            tree_rule.write(r)

        tree_features = set([feats_labels[i] for i in clf.tree_.feature if i >= 0])
        print(tree_features)
        print(len(tree_features))
        with open('{}tree/{}_features.txt'.format(out_fold, fold+1), 'w') as tree_f:
            for f in list(tree_features):
                tree_f.write(f+'\n')

        ranks_values.append(list(tree_features))
        ranks_labels.append(fold+1)

        for dataset in l:
            print('### {}:'.format(dataset[1]))

            C = dataset[0][cfg.class_label].nunique()
            D = len(dataset[0].columns) - 1
            N = len(df.index)
            print('\n{} classes, {} features, {} samples\n'.format(C, D, N)) 

            plot_pca.plot(df, features=list(tree_features), norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}tree/{}_{}_{}.png'.format(out_fold, fold+1, dataset[1], cfg.viz_method), method=cfg.viz_method, task=cfg.task)

    venn_df = RR_utils.venn(ranks_values, ranks_labels, len(feats_labels))
    print(venn_df)
    venn_df.to_csv('{}tree/cv_venn.csv'.format(out_fold))

    if cfg.task == 'classification':
        if cfg.eval_metric == 'accuracy':
            RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, '{}tree/acc.txt'.format(out_fold))
        elif cfg.eval_metric == 'f1score':
            RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, '{}tree/f1score.txt'.format(out_fold))
        else:
            raise Exception('Unknown evaluation metric: {}'.format(cfg.eval_metric))       
    elif cfg.task == 'regression':
        RR_utils.save_accuracy(tr_accuracy, te_accuracy, cfg.k, '{}tree/mse.txt'.format(out_fold))
