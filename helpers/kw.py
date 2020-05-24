# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy import stats

import RR_utils
import plot_pca

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

def run_kw(df, selection, class_label, save_file):
    print(dataset)
    labels = list(np.sort(df[class_label].astype(str).unique()))

    all_features = list(df.columns)
    all_features.remove(class_label)
    pvs = []
    i=0
    for feat in all_features:
        conditions = []
        for label in labels:
            fdf = df[df[class_label]==label][feat]
            conditions.append(fdf.values)
        try:
            pv = stats.kruskal(*conditions)
            pvs.append((feat, pv.pvalue))
            i=i+1
            print(i, feat, pv.pvalue)
        except ValueError:
            pvs.append((feat, 1.0))
            i=i+1
            print(i, feat, 1.0)
    pvs.sort(key = lambda x: x[1])
    return [x[0] for x in pvs][0:selection] 

if __name__ == '__main__': 
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))
    splits = spt.splits

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=0)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    ranks_values = []
    ranks_labels = []
    for fold in range(len(splits)):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        #out = pd.read_csv('{}_{}_out.csv'.format(out_file, fold+1), delimiter=cfg.dataset_sep, header=0, index_col=0)
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled)
        print(tr_df)

        if te_df.empty:
            l = [(tr_df, 'train')]
        else:
            l = [(tr_df, 'train'), (te_df, 'test')]

        for dataset in l:
            print('\n### {}:'.format(dataset[1]))

            C = dataset[0][cfg.class_label].nunique()
            D = len(dataset[0].columns)-1
            N = len(df.index)
            print('\n{} classes, {} features, {} samples\n'.format(C, D, N)) 

            if not os.path.exists(out_fold + 'kw/'):
                os.makedirs(out_fold + 'kw/')
            kw_datafile = '{}kw/{}_{}.csv'.format(out_fold, fold+1, dataset[1])

            kw_features = run_kw(dataset[0], selection=max(50, cfg.n_selection), class_label=cfg.class_label, save_file='{}kw/{}_{}_selection_complete.txt'.format(out_fold, fold+1, dataset[1]))
           
            with open('{}kw/{}_{}_selection.txt'.format(out_fold, fold+1, dataset[1]), 'w') as sf:
                for feat in kw_features[0:cfg.n_selection]:
                    sf.write(feat + '\n')
            plot_pca.plot(df, features=kw_features[0:cfg.n_selection], norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name=kw_datafile.replace('.csv', '_{}.png'.format(cfg.viz_method)), method=cfg.viz_method)
            ranks_values.append(kw_features)
            ranks_labels.append('{}_{}'.format(fold+1, dataset[1]))
    
    tau_df = RR_utils.kendall_tau(ranks_values, ranks_labels, cfg.n_selection)
    print(tau_df)
    tau_df.to_csv('{}kw/cv_dist.csv'.format(out_fold))

    venn_df = RR_utils.venn(ranks_values, ranks_labels, cfg.n_selection)
    print(venn_df)
    venn_df.to_csv('{}kw/cv_venn.csv'.format(out_fold))
