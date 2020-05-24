# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

def compute_distance(df, class_label, label, dist):
    sel = df[df[class_label] == label]
    sel = sel.drop([class_label], axis=1).values
    if dist == 'intra':
        d = pdist(sel, 'euclidean')
    elif dist == 'inter':
        not_sel = df[df[cfg.class_label] != cl]
        not_sel = not_sel.drop([class_label], axis=1).values
        d = []
        for a in sel:
            for b in not_sel:
                d.append(np.linalg.norm(a-b))
        d = np.array(d)
    else:
        raise Exception('Unknown dist type: {}'.format(dist))
    return d

def dataframe_avg(dataframes, filename):
    # https://stackoverflow.com/questions/38940946/average-of-multiple-dataframes-with-the-same-columns-and-indices
    avg = pd.concat(dataframes).groupby(level=0).mean()
    sd  = pd.concat(dataframes).groupby(level=0).std()
    avg = avg.reindex(columns=dataframes[0].columns, index=dataframes[0].index)
    sd  = sd.reindex(columns=dataframes[0].columns, index=dataframes[0].index)
    avg.to_csv(filename)
    sd.to_csv(filename.replace('.csv', '_std.csv'))
    print('\n Average distance: {}\n'.format(filename))
    print(avg)

if __name__ == '__main__': 
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    class_labels = list(np.sort(pd.read_csv(cfg.dataset_file, delimiter=',', header=0, index_col=0)[cfg.class_label].unique()))
    class_labels = [str(cl) for cl in class_labels]   

    all_folds_train_intra = []
    all_folds_train_inter = []
    all_folds_test_intra  = []
    all_folds_test_inter  = []

    for fold in range(1, cfg.k+1):
        for use in ['train', 'test']:
            print('\n', fold, use, '\n')
            file_names = []
            avg_intra_dist = {}
            avg_inter_dist = {}
            for cl in class_labels:
                avg_intra_dist[cl] = [] 
                avg_inter_dist[cl] = [] 
            
            for file_label in class_labels + ['aver', 'combine', 'ww', 'mrmr', 'kw', 'tree', 'original']:
                if file_label == 'original':
                    file = '{}_{}.csv'.format(out_file, cfg.viz_method)
                elif file_label == 'ww':
                    file = '{}_{}_{}_{}_{}.csv'.format(out_file.replace(out_fold, out_fold+'ww/'), fold, use, 'aver', cfg.viz_method)
                elif file_label == 'mrmr':
                    file = '{}mrmr/{}_{}_{}.csv'.format(out_fold, fold, use, cfg.viz_method)
                elif file_label == 'kw':
                    file = '{}kw/{}_{}_{}.csv'.format(out_fold, fold, use, cfg.viz_method)
                elif file_label == 'tree':
                    file = '{}tree/{}_{}_{}.csv'.format(out_fold, fold, use, cfg.viz_method)
                else:
                    file = '{}_{}_{}_{}_{}.csv'.format(out_file, fold, use, file_label, cfg.viz_method)

                try:
                    df = pd.read_csv(file, delimiter=',', header=0, index_col=0)
                    df.columns = df.columns.str.strip()
                    df[cfg.class_label] = df[cfg.class_label].astype(str).str.strip()

                    for cl in class_labels:
                        intra_dist = compute_distance(df, cfg.class_label, cl, 'intra')
                        avg_intra_dist[cl].append(intra_dist.mean())
                        inter_dist = compute_distance(df, cfg.class_label, cl, 'inter')
                        avg_inter_dist[cl].append(inter_dist.mean())

                    print(file)
                    file_names.append(file_label)

                except:
                    print('Could not open file {}'.format(file))

            intra_dist_df = pd.DataFrame(avg_intra_dist, index=file_names)
            print('\nDistance intraclass:\n')
            print(intra_dist_df)
            print('\nDistance interclass:\n')
            intra_dist_df.to_csv('{}{}_{}_intra_dist.csv'.format(out_fold, fold, use))

            inter_dist_df = pd.DataFrame(avg_inter_dist, index=file_names)
            print(inter_dist_df)
            inter_dist_df.to_csv('{}{}_{}_inter_dist.csv'.format(out_fold, fold, use))

            if use == 'train':
                all_folds_train_intra.append(intra_dist_df)
                all_folds_train_inter.append(inter_dist_df)
            elif use == 'test':
                all_folds_test_intra.append(intra_dist_df)
                all_folds_test_inter.append(inter_dist_df)

    print('\n###\n')
    dataframe_avg(all_folds_train_intra, '{}train_intra_dist.csv'.format(out_fold))
    dataframe_avg(all_folds_train_inter, '{}train_inter_dist.csv'.format(out_fold))
    dataframe_avg(all_folds_test_intra, '{}test_intra_dist.csv'.format(out_fold))
    dataframe_avg(all_folds_test_inter, '{}test_inter_dist.csv'.format(out_fold))