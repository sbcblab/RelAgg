# Bruno Iochins Grisci
# October 17th, 2021

import os
import sys
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import RR_utils

def main():
    config_file = sys.argv[1]
    class_col   = sys.argv[2]
    cv_fold     = sys.argv[3]
    fold_type   = sys.argv[4]

    cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))
    
    print('Loading ' + cfg.dataset_file)
    # https://stackoverflow.com/questions/24962908/how-can-i-read-only-the-header-column-of-a-csv-file-using-python
    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index, nrows=1)
    # df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    df.drop([cfg.class_label], axis=1, inplace=True)

    print(df)
    
    cp_list = list(range(cfg.checkpoint, cfg.train_epochs+1, cfg.checkpoint))
    if cp_list[-1] != cfg.train_epochs:
        cp_list.append(cfg.train_epochs)
    
    for cp in cp_list: 

        print('Loading ' + cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + os.path.basename(cfg.dataset_file).replace('.csv', '_{}_{:04d}_{}_datasheet.csv'.format(cv_fold, cp, fold_type)))
        relevance = pd.concat([chunk for chunk in tqdm(pd.read_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + os.path.basename(cfg.dataset_file).replace('.csv', '_{}_{:04d}_{}_datasheet.csv'.format(cv_fold, cp, fold_type)), delimiter=',', header=0, index_col=0, low_memory=False, chunksize=cfg.load_chunksize), desc='Loading data from {} in chunks of {}'.format(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + os.path.basename(cfg.dataset_file).replace('.csv', '_{}_{:04d}_{}_datasheet.csv'.format(cv_fold, cp, fold_type)), cfg.load_chunksize))])
        #relevance = relevance.astype({c: cfg.dtype_float for c in relevance.select_dtypes(include='float64').columns})
        #relevance = relevance.astype({c: cfg.dtype_int for c in relevance.select_dtypes(include='int64').columns})

        relevance.rename(columns={class_col:'value'}, inplace=True)
        print(relevance)
        scores = relevance['value']
        del relevance
        scores.dropna(inplace=True)
        
        scores.index = scores.index.map(str)
        df.columns = df.columns.map(str)
        
        print('Scores:')
        print(scores)
        print('NaN:')
        print(scores.isna().sum())
        print('df.columns:')
        print(df.columns)
        print('scores.index:')
        print(scores.index)
        scores = scores.reindex(index=df.columns)
        scores.index.name = 'feature'
        print('Scores:')
        print(scores)
        print('NaN:')
        print(scores.isna().sum())       
        print('mean, std:')
        print(scores.mean(), scores.std())
        print('median:')
        print(np.median(scores))
        print('min, max:')
        print(scores.min(), scores.max())
        print('Writing ' + cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + 'RelAgg_{:04d}_'.format(cp) + class_col + '_' + os.path.basename(cfg.dataset_file))
        scores.to_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + 'RelAgg_{:04d}_'.format(cp) + class_col + '_' + os.path.basename(cfg.dataset_file))
        del scores
    del df

if __name__ == '__main__': 
    main()