# Bruno Iochins Grisci
# October 17th, 2021

import os
import sys
import importlib
import numpy as np
import pandas as pd

import RR_utils

def main():
    config_file = sys.argv[1]
    cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))
    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    relevance = pd.read_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + os.path.basename(cfg.dataset_file).replace('.csv', '_1_train_datasheet.csv'), delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)

    print(df)
    relevance.rename(columns={'score':'value'}, inplace=True)
    print(relevance)
    scores = relevance['value']
    scores.dropna(inplace=True)

    df.drop([cfg.class_label], axis=1, inplace=True)
    print(scores)
    print(df.columns)
    print(scores.index)

    scores = scores.reindex(df.columns)
    scores.index.name = 'feature'
    print(scores)
    scores.to_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + 'RelAgg_' + os.path.basename(cfg.dataset_file))

if __name__ == '__main__': 
    main()