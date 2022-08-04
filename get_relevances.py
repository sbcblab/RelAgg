# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import numpy as np ; na = np.newaxis
from scipy.stats import gmean
import pandas as pd
import importlib
import importlib.util
from collections import namedtuple
from tensorflow.keras.models import load_model
from tqdm import tqdm

import model_io
import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

if cfg.rel_rule[0] == 'deeplift':
    import deeplift
    from deeplift.conversion import kerasapi_conversion as kc

SCORE_LABEL = 'score'

##########################################################################################

def compute_lift(df, nn_lift, model, task_type, class_label, rel_class, rule_param, batch_size, file_name):
    
    if task_type == 'regression' and rel_class != 'pred':
        print('\nWARNING: For regression "rel_class" must be "pred". Value changed from "{}".\n'.format(rel_class))
        rel_class = 'pred'

    find_scores_layer_idx = 0
    if task_type == 'classification':
        target_layer_idx = -2
    elif task_type == 'regression':
        target_layer_idx = -1
    deeplift_contribs_func = nn_lift.get_target_contribs_func(
                             find_scores_layer_idx=find_scores_layer_idx,
                             target_layer_idx=target_layer_idx) #For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
                                                                #For regression tasks with a linear output, target_layer_idx should be -1

    X, Y = RR_utils.get_XY(df, task_type, class_label)

    if len(rule_param) == 1:
        reference = np.zeros(X.shape)
        print('WARNING: No reference value for DeepLIFT, will use zero.')
    elif rule_param[1] == 'mean':
        #reference = np.tile(X.mean(axis=0), X.shape)
        reference = X.mean(axis=0)
        reference = np.tile(reference,(X.shape[0],1))
        print(X.shape)
        print(reference.shape)
    elif rule_param[1] == 'zero' or rule_param[1] == 0.0:
        reference = np.zeros(X.shape)
    elif isinstance(rule_param[1], float) or isinstance(rule_param[1], int):
        reference = np.ones(X.shape)
        reference = reference * rule_param[1]
        print(X.shape)
        print(reference.shape)
    else:
        raise Exception('Wrong value for the reference of DeepLIFT: {}'.format(rule_param[1]))

    print('Computing DeepLIFT')
    if task_type == 'classification' and (rel_class == 'true' or rel_class == 'pred'):
        lift = []
        if rel_class == 'true':
            ypart = Y
        elif rel_class == 'pred':
            ypart = model.predict(X)
        partition = np.where(np.diff(np.argmax(ypart, axis=1)) != 0)[0]+1
        print(len(partition), partition)
        for x, ref, y in zip(np.split(X, partition), np.split(reference, partition), np.split(ypart, partition)):
            if x.size != 0:
                task = np.argmax(y[0])
                lift.append(np.array(deeplift_contribs_func(task_idx=task,
                                                            input_data_list=[x],
                                                            input_references_list=[ref],
                                                            batch_size=batch_size,
                                                            progress_update=1000)))
        lift = np.concatenate(lift, axis=0) 

    elif task_type == 'regression' or (task_type == 'classification' and rel_class >= 0 and rel_class < len(Y[0])):
        partition = 10
        if task_type == 'regression':
            task = 0
        elif rel_class >= 0 and rel_class < len(Y[0]):
            task = rel_class
        else:
            raise Exception('Wrong value for the relevance class: {}'.format(rel_class))
        lift = []
        print(partition)
        complete_i = 0
        for x, ref in zip(np.array_split(X, partition), np.array_split(reference, partition)):
            print('Computing relevance: {:.2f}%'.format(complete_i/float(len(Y))*100), end='\r')
            complete_i += partition            
            if x.size != 0:
                lift.append(np.array(deeplift_contribs_func(task_idx=task,
                                                            input_data_list=[x],
                                                            input_references_list=[ref],
                                                            batch_size=batch_size,
                                                            progress_update=1000)))
        lift = np.concatenate(lift, axis=0)

    else:
        raise Exception('Wrong value for the relevance class: {}'.format(rel_class))

    print(X.shape, lift.shape)

    list_wo_y = list(df.columns.values)
    for cl in class_label:
        list_wo_y.remove(cl)
    relevance = pd.DataFrame(lift, index=list(df.index.values), columns=list_wo_y)
    if file_name:
        print('Writing ' + file_name)
        RR_utils.write_large_csv(relevance, file_name)
        #relevance.to_csv(file_name)   
    
    del X
    del Y
    del list_wo_y
    return relevance

def compute_relevance(df, nn, task, class_label, rel_class, lrp_param, file_name):
    
    if task == 'regression' and rel_class != 'pred':
        print('\nWARNING: For regression "rel_class" must be "pred". Value changed from "{}".\n'.format(rel_class))
        rel_class = 'pred'

    if len(lrp_param) > 1:
        if lrp_param[1] is not None:
            nn.set_lrp_parameters(lrp_param[0], lrp_param[1])
        else:
            nn.set_lrp_parameters(lrp_param[0])
    else:
        nn.set_lrp_parameters(lrp_param[0])

    # set the first layer (a convolutional layer) decomposition variant to 'w^2'. This may be especially
    # usefull if input values are ranged [0 V], with 0 being a frequent occurrence
    # the result with display relevance in important areas despite zero input activation energy.

    nn.modules[0].set_lrp_parameters('ww') # also try 'flat'
    # compute the relevance map

    X, Y = RR_utils.get_XY(df, task, class_label)
    R = []
    partition = 100
    complete_i = 0
    for x, y in zip(np.array_split(X, partition), np.array_split(Y, partition)):
        print('Computing relevance: {:.2f}%'.format(complete_i/float(len(Y))*100), end='\r')
        complete_i += partition
        if x.size != 0:
            ypred = nn.forward(x) + 0.0001 # sum small error factor to network output because initial relevance must be > 0
            if rel_class == 'true':
                Rinit = ypred*y
            elif rel_class == 'pred':
                #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
                mask = np.zeros_like(ypred)
                mask[np.arange(len(ypred)), ypred.argmax(axis=1)] = 1
                Rinit = ypred*mask
            elif rel_class >= 0 and rel_class < y.shape[1]:
                yselect = (np.arange(y.shape[1])[na,:] == rel_class)*1.0
                yselect = np.repeat(yselect, y.shape[0], axis=0)
                Rinit = ypred*yselect
            else:
                raise Exception('Wrong value for the relevance class.')
            R.append(nn.lrp(Rinit))
            del Rinit
            del ypred

    R = np.concatenate(R,axis=0)
    print(X.shape, R.shape)

    list_wo_y = list(df.columns.values)
    for cl in class_label:
        list_wo_y.remove(cl)
    relevance = pd.DataFrame(R, index=list(df.index.values), columns=list_wo_y)
    
    del R
    del X
    del Y
    del list_wo_y
    
    if file_name:
        print('Writing ' + file_name)
        RR_utils.write_large_csv(relevance, file_name)
        # relevance.to_csv(file_name)
    return relevance

def compute_rel_stats(df, file_name):
    stats = pd.DataFrame()
    stats['sum']    = df.sum(axis=0)
    stats['aver']   = df.mean(axis=0)
    stats['std']    = df.std(axis=0)
    stats['median'] = df.median(axis=0)
    stats['min']    = df.min(axis=0)
    stats['max']    = df.max(axis=0)
    abs_df = df.abs()
    stats['abs_sum']    = abs_df.sum(axis=0)
    stats['abs_aver']   = abs_df.mean(axis=0)
    stats['abs_std']    = abs_df.std(axis=0)
    stats['abs_median'] = abs_df.median(axis=0)
    stats['abs_min']    = abs_df.min(axis=0)
    stats['abs_max']    = abs_df.max(axis=0)
    print('# Stats')
    print(stats)
    print('Writing ' + file_name)
    stats.to_csv(file_name)
    del stats
    del abs_df

def agglutinate(df):
    cols = df.columns.values
    agg_cols = [c for c in cols if '.' in c]
    names = {}
    for ac in agg_cols:
        s = ac.split('.')[0]
        if s in names:
            names[s].append(ac)
        else:
            names[s] = [ac]

    for name in names:
        sum_name  = df[names[name]].sum(axis=1) / len(names[name])
        df = df.drop(names[name], axis=1)
        df[name] = sum_name

    del cols
    del agg_cols
    del names
    return df

def rank(rel, y, rank_metric, mean_type, class_labels, file_name):
    if rank_metric == 'rank':
        ranking = rel.abs().rank(axis=1, ascending=False)
    elif rank_metric == 'norm':
        ranking = rel.abs().div(rel.abs().max(axis=1), axis=0)
        ranking = ranking.fillna(0.0)
    else:
        raise Exception('Unknown rank method: {}'.format(rank_metric))
    rankingT = ranking.T

    # https://stackoverflow.com/questions/39658574/how-to-drop-columns-which-have-same-values-in-all-rows-via-pandas-or-spark-dataf
    aver_func = None
    if mean_type == 'arithmetic':
        aver_func = np.mean
    elif mean_type == 'geometric':
        aver_func = gmean
    else:
        raise Exception('Unknown mean type: {}'.format(mean_type))  

    for cl in class_labels:
        if y is not None and len(class_labels) > 1:
            rwot = ranking[y==cl].T
        else:
            rwot = ranking.T
        rwot = rwot.drop(rwot.std()[(rwot.std() == 0)].index, axis=1)
        rankingT[cl] = np.nan_to_num(aver_func(rwot, axis=1), nan=0.0)
        del rwot
    class_ranks = rankingT[class_labels].values
    rankingT.insert(rankingT.columns.get_loc(class_labels[0]), SCORE_LABEL, aver_func(class_ranks, axis=1), True)
    #rankingT.insert(rankingT.columns.get_loc(class_labels[0]), SCORE_LABEL, (class_ranks.min(axis=1) + class_ranks.max(axis=1))/2.0, True)
    #rankingT.insert(rankingT.columns.get_loc(class_labels[0]), SCORE_LABEL, class_ranks.max(axis=1), True)
  
    rankingT = rankingT.fillna(0.0)
    if rank_metric == 'rank':
        #rankingT = rankingT.sort_values(SCORE_LABEL)
        rankingT.sort_values(SCORE_LABEL, inplace=True)
    elif rank_metric == 'norm':
        #rankingT = rankingT.sort_values(SCORE_LABEL, ascending=False)
        rankingT.sort_values(SCORE_LABEL, ascending=False, inplace=True)
    print('Writing ' + file_name)
    RR_utils.write_large_csv(rankingT, file_name)
    #rankingT.to_csv(file_name)
    del ranking
    return rankingT

def rank_dist(ranking, y, n_features, list_classes, file_name):
    labels = []
    ranks  = []
    list_classes = list(list_classes)
    for cl in [SCORE_LABEL] + list_classes:
        labels.append(cl)
        rk = ranking.sort_values(cl)
        ranks.append(rk.index.values)
    tau_df = RR_utils.kendall_tau(ranks, labels, n_features)
    print('Writing ' + file_name)
    tau_df.to_csv(file_name)
    del tau_df

def rank_venn(ranking, y, n_features, list_classes, file_name):
    labels = []
    ranks  = []
    list_classes = list(list_classes)
    for cl in [SCORE_LABEL] + list_classes:
        labels.append(cl)
        rk = ranking.sort_values(cl)
        ranks.append(rk.index.values)
    venn_df = RR_utils.venn(ranks, labels, n_features)
    print('Writing ' + file_name)
    venn_df.to_csv(file_name)
    del venn_df

def heatsheets(data, ranking, out, ys, class_label, use, agg, file_name):
    if agg:
        print('Agglutinating...')
        rel_index = list(ranking.index)
        dat_index = list(data.columns)
        if class_label in dat_index:
            dat_index.remove(class_label)
        for ri in rel_index:
            if ri not in dat_index:
                rif = [f for f in dat_index if ri in f]     
                #mif = data[rif].idxmax(axis=1)
                # https://stackoverflow.com/questions/14734695/get-column-name-where-value-is-something-in-pandas-dataframe
                #mif = data[rif].apply(lambda x: data[rif].columns[x==x.max()], axis = 1)
                mif = data[rif].apply(lambda x: data[rif].columns[x>0], axis = 1)
                mif = mif.apply(lambda x: ''.join(x.values))
                mif = mif.str.replace(ri+'.', '')
                data[ri] = mif
                data = data.drop(rif, axis=1)
                del rif
                del mif
        del rel_index
        del dat_index

    print(out)
    out_part = out.loc[out['usage'] == use].copy()
    out_part = out_part.T
    for label in [SCORE_LABEL] + list(ys):
        out_part[label] = None
    print([SCORE_LABEL] + list(ys))
    print('Sorting...')
    out_part.sort_values(by=['usage', class_label, 'prediction', 'max_out'], axis=1, ascending=True, na_position='first', inplace=True)
    print(out_part)
    
    rank_sum = ranking[ranking.columns[-(1+len(ys)):]].copy()
    save_size_limit = 20_000_000 
    if data.size < save_size_limit:
        rank_sum = rank_sum.join((data.T).reindex(rank_sum.index))
    else:
        print('WARNING: Data has size {} larger than {}, so it was not saved in {}'.format(data.size, save_size_limit, file_name))
    rank_sum = out_part.append(rank_sum, sort=True)
    rank_sum = rank_sum[out_part.columns]
    del out_part
    print(rank_sum)
    print('Writing ' + file_name)
    RR_utils.write_large_csv(rank_sum, file_name)
    #rank_sum.to_csv(file_name)
    del rank_sum   
    
#################################################################################################3

if __name__ == '__main__':

    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:    
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    #spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))

    print('Loading ' + spt_file)
    spec = importlib.util.spec_from_file_location(spt_file.replace('/','.').replace('.py',''), spt_file) 
    spt  = importlib.util.module_from_spec(spec) 
    spec.loader.exec_module(spt) 

    splits = spt.splits

    if not os.path.exists(out_fold+'relevance_eval/'):
        os.makedirs(out_fold+'relevance_eval/')

    print('Loading ' + cfg.dataset_file)
    
    # https://stackoverflow.com/questions/52209290/how-do-i-make-a-progress-bar-for-loading-pandas-dataframe-from-a-large-xlsx-file
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index, chunksize=cfg.load_chunksize), desc='Loading data from {} in chunks of {}'.format(cfg.dataset_file, cfg.load_chunksize))])
    #data = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index, chunksize=1000)
    #df = pd.concat(data)
    #del data
    
    df = RR_utils.set_dataframe_dtype(df, cfg.dtype_float, cfg.dtype_int)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    sort_class = []
    if cfg.task == 'classification':
        sort_class = np.sort(df[cfg.class_label].unique())
    elif cfg.task == 'regression':
        sort_class, target_classes = RR_utils.split_targets(df, df[cfg.class_label].astype(float).min(), df[cfg.class_label].astype(float).max(), cfg.target_split, cfg.class_label)
    print(sort_class)

    if cfg.task == 'regression':
        df[cfg.class_label] = RR_utils.shift_target(df, cfg.class_label)

    ranks_values = []
    ranks_labels = []

    cp_list = list(range(cfg.checkpoint, cfg.train_epochs+1, cfg.checkpoint))
    if cp_list[-1] != cfg.train_epochs:
        cp_list.append(cfg.train_epochs)

    for fold in range(len(splits)):
        
        print('\n###### {}-FOLD:\n'.format(fold+1))
        
        for cp in cp_list: 

            print('Loading ' + '{}_{}_{:04d}_out.csv'.format(out_file, fold+1, cp))
            out = pd.concat([chunk for chunk in tqdm(pd.read_csv('{}_{}_{:04d}_out.csv'.format(out_file, fold+1, cp), delimiter=',', header=0, index_col=0, chunksize=cfg.load_chunksize), desc='Loading data from {} in chunks of {}'.format(cfg.dataset_file, cfg.load_chunksize))])
            #data_out = pd.read_csv('{}_{}_{:04d}_out.csv'.format(out_file, fold+1, cp), delimiter=',', header=0, index_col=0, chunksize=1000)
            #out = pd.concat(data_out)
            #del data_out
            out = out.astype({c: cfg.dtype_float for c in out.select_dtypes(include='float64').columns})
            
            #load a neural network
            print('# Reading neural network {:04d}'.format(cp))
            print('Loading ' + '{}_{}_{:04d}.hdf5'.format(out_file, fold+1, cp))
            model = load_model('{}_{}_{:04d}.hdf5'.format(out_file, fold+1, cp), custom_objects={'NonPos': RR_utils.NonPos, 'IsZero': RR_utils.IsZero}, compile=False)
            if cfg.rel_rule[0] == 'deeplift':
                nn_lift =\
                          kc.convert_model_from_saved_files(
                          '{}_{}_{:04d}.hdf5'.format(out_file, fold+1, cp),
                          nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
                          #nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.Rescale)   
            else:
                nn_file = '{}_{}_{:04d}.txt'.format(out_file, fold+1, cp)
                RR_utils.write_model(model, nn_file)
                nn = model_io.read(nn_file)           

            tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled) 
            tr_df2, te_df2, mean_vals2, std_vals2, min_vals2, max_vals2 = RR_utils.split_data(df, splits[fold], cfg.class_label, False, False) 
            
            tr_df = tr_df.astype({c: cfg.dtype_float for c in tr_df.select_dtypes(include='float64').columns})
            te_df = te_df.astype({c: cfg.dtype_float for c in te_df.select_dtypes(include='float64').columns})
            tr_df2 = tr_df2.astype({c: cfg.dtype_float for c in tr_df2.select_dtypes(include='float64').columns})
            te_df2 = te_df2.astype({c: cfg.dtype_float for c in te_df2.select_dtypes(include='float64').columns})
            
            tr_df = tr_df.astype({c: cfg.dtype_int for c in tr_df.select_dtypes(include='int64').columns})
            te_df = te_df.astype({c: cfg.dtype_int for c in te_df.select_dtypes(include='int64').columns})
            tr_df2 = tr_df2.astype({c: cfg.dtype_int for c in tr_df2.select_dtypes(include='int64').columns})
            te_df2 = te_df2.astype({c: cfg.dtype_int for c in te_df2.select_dtypes(include='int64').columns})            
            
            del mean_vals
            del std_vals
            del min_vals
            del max_vals
            del mean_vals2
            del std_vals2
            del min_vals2
            del max_vals2

            if te_df.empty:
                l = [(tr_df, 'train', tr_df2)]
            else:
                l = [(tr_df, 'train', tr_df2), (te_df, 'test', te_df2)]
            for dataset in l:
                print('### {}:'.format(dataset[1]))

                print('Computing relevance...')
                if cfg.rel_rule[0] != 'deeplift':
                    rel = compute_relevance(dataset[0], nn, cfg.task, [cfg.class_label], cfg.rel_class, cfg.rel_rule, '{}_{}_{:04d}_{}_relevance.csv'.format(out_file, fold+1, cp, dataset[1]))
                else:
                    rel = compute_lift(dataset[0], nn_lift, model, cfg.task, [cfg.class_label], cfg.rel_class, cfg.rel_rule, cfg.batch_size, '{}_{}_{:04d}_{}_relevance.csv'.format(out_file, fold+1, cp, dataset[1]))

                rel = rel.astype({c: cfg.dtype_float for c in rel.select_dtypes(include='float64').columns})    
                    
                if cfg.agglutinate:
                    print('Agglutinating...')
                    rel = agglutinate(rel)

                for cl in sort_class:
                    if cl in list(rel.index):
                        rel.rename(index={cl:'{}_'.format(cl)},inplace=True)
                        dataset[0].rename(index={cl:'{}_'.format(cl)},inplace=True)
                        dataset[2].rename(index={cl:'{}_'.format(cl)},inplace=True)
                        print('\nWARNING: index {} was changed to {}_ due to name collision with a class label.\n'.format(cl, cl))

                print('Computing relevance stats...')
                compute_rel_stats(rel, '{}_{}_{:04d}_{}_rel_stats.csv'.format(out_fold+'relevance_eval/', fold+1, cp, dataset[1]))
                
                print('Computing ranking...')
                if cfg.task == 'classification':
                    ranking = rank(rel, dataset[0][cfg.class_label], cfg.rank, cfg.mean_type, sort_class, '{}{}_{:04d}_{}_rank.csv'.format(out_fold+'relevance_eval/', fold+1, cp, dataset[1]))
                elif cfg.task == 'regression':
                    tc = target_classes.loc[rel.index]
                    ranking = rank(rel, tc[0], cfg.rank, cfg.mean_type, sort_class, '{}{}_{:04d}_{}_rank.csv'.format(out_fold+'relevance_eval/', fold+1, cp, dataset[1]))
                ranking = ranking.astype({c: cfg.dtype_float for c in ranking.select_dtypes(include='float64').columns})
                    
                print('# Relevance')
                print(rel)

                print('# Ranking')
                print(ranking)
                ranks_values.append(ranking.index.values)
                ranks_labels.append('{}_{}'.format(fold+1, dataset[1]))

                if cfg.kendall_tau:
                    print('Computing kendall tau...')
                    rank_dist(ranking, dataset[0][cfg.class_label], cfg.n_selection, sort_class, '{}{}_{:04d}_{}_rank_dist.csv'.format(out_fold+'relevance_eval/', fold+1, cp, dataset[1]))
                if cfg.venn:
                    print('Computing venn diagram...')
                    rank_venn(ranking, dataset[0][cfg.class_label], cfg.n_selection, sort_class, '{}{}_{:04d}_{}_rank_venn.csv'.format(out_fold+'relevance_eval/', fold+1, cp, dataset[1]))
                print(out)
                print('Computing relevance heatmap...')
                heatsheets(rel, ranking, out, sort_class, cfg.class_label, dataset[1], cfg.agglutinate, '{}_{}_{:04d}_{}_relsheet.csv'.format(out_file, fold+1, cp, dataset[1]))
                del rel
                print('Computing data heatmap...')
                heatsheets(dataset[2], ranking, out, sort_class, cfg.class_label, dataset[1], cfg.agglutinate, '{}_{}_{:04d}_{}_datasheet.csv'.format(out_file, fold+1, cp, dataset[1]))
                del ranking 
            
            del tr_df
            del te_df
            del tr_df2
            del te_df2
            del l            
            del out
            del model
            if cfg.rel_rule[0] == 'deeplift':
                del nn_lift
            else:
                del nn 
    
    del df
    
    if cfg.kendall_tau:
        print('Computing kendall tau...')
        tau_df = RR_utils.kendall_tau(ranks_values, ranks_labels, cfg.n_selection)
        print(tau_df)
        print('Writing ' + '{}cv_dist.csv'.format(out_fold+'relevance_eval/'))
        tau_df.to_csv('{}cv_dist.csv'.format(out_fold+'relevance_eval/'))
        del tau_df

    if cfg.venn:
        print('Computing venn diagram...')
        venn_df = RR_utils.venn(ranks_values, ranks_labels, cfg.n_selection)
        print(venn_df)
        print('Writing ' + '{}cv_venn.csv'.format(out_fold+'relevance_eval/'))
        venn_df.to_csv('{}cv_venn.csv'.format(out_fold+'relevance_eval/'))    
        del venn_df