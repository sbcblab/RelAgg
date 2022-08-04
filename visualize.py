# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import importlib
import importlib.util
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.stats.mstats import rankdata 

import RR_utils
import plot_pca

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

colhex = {
    'RED':     'BA0000',
    'BLUE':    '0000FF',
    'YELLOW':  'FFEE00',
    'GREEN':   '048200',    
    'ORANGE':  'FF6103',
    'BLACK':   '000000',
    'CYAN':    '00FFD4',    
    'SILVER':  'c0c0c0',
    'MAGENTA': '680082',
    'CREAM':   'FFFDD0',
    'DRKBRW':  '654321',
    'BEIGE':   'C2C237',
    'WHITE':   'FFFFFF',
}

TRAINTEST = ['train', 'test']
SCORE_LABEL = 'score'

ONLY_WEIGHTED = False

####################################################################################

if __name__ == '__main__':

    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:    
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    #spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))
    spec = importlib.util.spec_from_file_location(spt_file.replace('/','.').replace('.py',''), spt_file) 
    spt  = importlib.util.module_from_spec(spec) 
    spec.loader.exec_module(spt) 
    splits = spt.splits

    if not os.path.exists(out_fold+'viz/'):
        os.makedirs(out_fold+'viz/')

    if not os.path.exists(out_fold+'viz/sel/'):
        os.makedirs(out_fold+'viz/sel/')        

    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index, chunksize=cfg.load_chunksize), desc='Loading data from {} in chunks of {}'.format(cfg.dataset_file, cfg.load_chunksize))])   
    df = RR_utils.set_dataframe_dtype(df, cfg.dtype_float, cfg.dtype_int)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)
    
    if cfg.task == 'classification':
        CLASS_LABELS = list(np.sort(df[cfg.class_label].astype(str).unique()))
    elif cfg.task == 'regression':
        CLASS_LABELS, _ = RR_utils.split_targets(df, df[cfg.class_label].astype(float).min(), df[cfg.class_label].astype(float).max(), cfg.target_split, cfg.class_label)
        CLASS_LABELS = list(CLASS_LABELS)
    else:
        raise Exception('Unknown task type: {}'.format(cfg.task))    
    n_classes  = len(CLASS_LABELS)
    n_features = len(df.columns.values) - 1

    if not ONLY_WEIGHTED:
        plot_pca.plot(df, norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}{}.pdf'.format(out_fold+'viz/', cfg.viz_method), method=cfg.viz_method, task=cfg.task, perplexity=cfg.perplexity, n_iter=cfg.n_iter)

    if cfg.k > 1:
        usage = TRAINTEST
    else:
        usage = TRAINTEST[0:1]

    for fold in range(len(splits)):
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, False, False)
        trte_df = {TRAINTEST[0]:df, TRAINTEST[1]:df}
        for use in usage:
            print('\n###### {}-FOLD: {}\n'.format(fold+1, use))

            if not ONLY_WEIGHTED:
                plot_pca.plot(trte_df[use], norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}{}_{}_{}.pdf'.format(out_fold+'viz/', fold+1, use, cfg.viz_method), method=cfg.viz_method, task=cfg.task, perplexity=cfg.perplexity, n_iter=cfg.n_iter)

            data_file = '{}_{}_{:04d}_{}_datasheet.csv'.format(out_file, fold+1, cfg.train_epochs, use)
            rele_file = '{}_{}_{:04d}_{}_relsheet.csv'.format(out_file, fold+1, cfg.train_epochs, use)

            dat = pd.read_csv(data_file, header=0, index_col=0, low_memory=False)
            rel = pd.read_csv(rele_file, header=0, index_col=0, low_memory=False)

            print('\nData:\n')
            print(dat)
            print('\nRelevance:\n')
            print(rel)

            df_selection = pd.DataFrame([], columns=[SCORE_LABEL] + CLASS_LABELS)
            if len(CLASS_LABELS) > 1:
                labels_to_viz = [SCORE_LABEL] + CLASS_LABELS
            else:
                labels_to_viz = [SCORE_LABEL]
            for cl in labels_to_viz:
                rank = dat[cl].copy()
                if cfg.rank == 'rank':
                    rank = rank.sort_values(ascending=True, na_position='last')
                elif cfg.rank == 'norm':
                    rank = rank.sort_values(ascending=False, na_position='last')
                else:
                    raise Exception('Unknown rank method: {}'.format(cfg.rank))
                rank_id = list(rank.index.values)[:cfg.n_selection]
                if cfg.agglutinate:
                    cat_rank_ids = []
                    for rid in rank_id:
                        if rid in list(df.columns.values):
                            cat_rank_ids.append(rid)
                        else:
                            for col in list(df.columns.values):
                                if rid + '.' in col:
                                    cat_rank_ids.append(col)
                    rank_id = cat_rank_ids
                df_selection[cl] = rank_id
                if not ONLY_WEIGHTED:
                    plot_pca.plot(trte_df[use], features=rank_id, norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}{}_{:04d}_{}_{}_{}.pdf'.format(out_fold+'viz/sel/', fold+1, cfg.train_epochs, use, cl, cfg.viz_method), method=cfg.viz_method, task=cfg.task, perplexity=cfg.perplexity, n_iter=cfg.n_iter)
                # trying something new:
                if not cfg.agglutinate:
                    if cfg.rank == "norm":
                        scores = dat[cl].copy()
                        scores.dropna(inplace=True)
                        new_index = list(trte_df[use].columns)
                        new_index.remove(cfg.class_label)
                        scores = scores.reindex(new_index)
                        plot_pca.plot(trte_df[use], norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}{}_{:04d}_{}_{}_{}.pdf'.format(out_fold+'viz/weighted_', fold+1, cfg.train_epochs, use, cl, cfg.viz_method), method='tsne', task=cfg.task, weights=scores, perplexity=cfg.perplexity, n_iter=cfg.n_iter)
                else:
                    print("\nWARNING: Can't show score weighted vizualization if agglutinate is true.")

            if len(CLASS_LABELS) > 1:
                all_ids = []
                i = 0
                while (len(all_ids) < cfg.n_selection) and i < cfg.n_selection:
                    for cl in CLASS_LABELS:
                        all_ids.append(df_selection[cl].values[i])
                        all_ids = list(set(all_ids))
                    i = i+1

                all_ids = list(set(all_ids))

                cat_all_ids = []
                if cfg.agglutinate:
                    for aid in all_ids:
                        if aid in list(df.columns.values):
                            cat_all_ids.append(aid)
                        else:
                            for col in list(df.columns.values):
                                if aid + '.' in col:
                                    cat_all_ids.append(col)
                else:
                    cat_all_ids = all_ids

                print('###\nCOMBINED SIZES: {}\n###'.format(len(all_ids)))
                if not ONLY_WEIGHTED:
                    plot_pca.plot(trte_df[use], features=cat_all_ids, norm=cfg.standardized, rescale=cfg.rescaled, class_label=cfg.class_label, colors=cfg.class_colors, file_name='{}{}_{:04d}_{}_{}_{}.pdf'.format(out_fold+'viz/sel/', fold+1, cfg.train_epochs, use, 'combine', cfg.viz_method), method=cfg.viz_method, task=cfg.task, perplexity=cfg.perplexity, n_iter=cfg.n_iter)