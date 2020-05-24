# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
from collections import namedtuple
import numpy as np
import pandas as pd

import RR_utils
import plot_pca

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

def df2mrmr(df, class_label, save_file):
    cols = list(df.columns)
    cols.remove(class_label)
    cols = [class_label] + cols
    df = df[cols]
    df=df.rename(columns = {class_label:'class'})
    df.to_csv(save_file, sep=',', index=False)

def str2df(str_list, token0, token1):
    str_list = str_list.split('\n')
    info = str_list[str_list.index(token0)+1:str_list.index(token1)]
    info = [line.strip().split('\t') for line in info if line.strip() != '']
    df = pd.DataFrame(info[1:], columns=info[0])
    df.columns = df.columns.str.strip()
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df

def mrmr(dataset_file, threshold=1, selection=50, method='MID', n_samples=1000, n_features=10000, save_file='mrmr.txt'):
    selection = min(selection, n_features)
    command = "./mrmr -i {} -t {} -n {} -m {} -s {} -v {} > {}".format(dataset_file, threshold, selection, method, n_samples, n_features, save_file)
    print('Running command:\n {}'.format(command))
    try:
        os.system(command)
    except:
        print("\nERROR: please download the binary mrmr file from http://home.penglab.com/proj/mRMR/ and place in the same directory as this script.")
    print('End.')
    maxrel_token = '*** MaxRel features ***'
    mrmr_token = '*** mRMR features *** '
    foot_token = ' *** This program and the respective minimum Redundancy Maximum Relevance (mRMR) '
    with open(save_file, 'r') as mrmr_file:
        cont = mrmr_file.read()

    maxrel_df = str2df(cont, maxrel_token, mrmr_token)
    mrmr_df   = str2df(cont, mrmr_token,   foot_token)

    return mrmr_df['Name'].values

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

    for c in df.columns:
        df = df.rename(columns = {c:"{}".format(c.replace(' ','').replace(',','').replace('/','').replace('\'', '').replace('"','').replace(':','').replace('-',''))})

    ranks_values = []
    ranks_labels = []
    for fold in range(len(splits)):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled)
        print(tr_df)

        if te_df.empty:
            l = [(tr_df, 'train')]
        else:
            l = [(tr_df, 'train'), (te_df, 'test')]

        for dataset in l:
            print('\n### {}:'.format(dataset[1]))

            C = dataset[0][cfg.class_label].nunique()
            D = len(dataset[0].columns) - 1
            N = len(df.index)
            print('\n{} classes, {} features, {} samples\n'.format(C, D, N)) 

            if not os.path.exists(out_fold + 'mrmr/'):
                os.makedirs(out_fold + 'mrmr/')
            mrmr_datafile = '{}mrmr/{}_{}.csv'.format(out_fold, fold+1, dataset[1])
            df2mrmr(dataset[0], cfg.class_label, mrmr_datafile)
            mrmr_features = mrmr(mrmr_datafile, selection=max(50, cfg.n_selection), n_samples=N, n_features=D, save_file='{}mrmr/{}_{}_selection_complete.txt'.format(out_fold, fold+1, dataset[1]))
            with open('{}mrmr/{}_{}_selection.txt'.format(out_fold, fold+1, dataset[1]), 'w') as sf:
                for feat in mrmr_features[0:cfg.n_selection]:
                    sf.write(feat + '\n')
            plot_pca.plot(df, features=mrmr_features[0:cfg.n_selection], norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name=mrmr_datafile.replace('.csv', '_{}.png'.format(cfg.viz_method)), method=cfg.viz_method)
            ranks_values.append(mrmr_features)
            ranks_labels.append('{}_{}'.format(fold+1, dataset[1]))
    
    tau_df = RR_utils.kendall_tau(ranks_values, ranks_labels, cfg.n_selection)
    print(tau_df)
    tau_df.to_csv('{}mrmr/cv_dist.csv'.format(out_fold))

    venn_df = RR_utils.venn(ranks_values, ranks_labels, cfg.n_selection)
    print(venn_df)
    venn_df.to_csv('{}mrmr/cv_venn.csv'.format(out_fold))
