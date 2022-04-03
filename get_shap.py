# Bruno Iochins Grisci
# May 5th, 2021

import os
import sys
import numpy as np ; na = np.newaxis
from scipy.stats import gmean
import pandas as pd
import importlib
import importlib.util
from collections import namedtuple
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt

import model_io
import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

if cfg.rel_rule[0] == 'deeplift':
    import deeplift
    from deeplift.conversion import kerasapi_conversion as kc

SCORE_LABEL = 'score'

##########################################################################################


if __name__ == '__main__':
    shap.initjs()
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

    if not os.path.exists(out_fold+'shap_eval/'):
        os.makedirs(out_fold+'shap_eval/')

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
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

    #for fold in range(len(splits)):
    for fold in range(1):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        out = pd.read_csv('{}_{}_out.csv'.format(out_file, fold+1), delimiter=',', header=0, index_col=0)
        #load a neural network

        print('# Reading neural network')
        model = load_model('{}_{}.hdf5'.format(out_file, fold+1), custom_objects={'NonPos': RR_utils.NonPos, 'IsZero': RR_utils.IsZero}, compile=False)
        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled) 
        tr_df2, te_df2, mean_vals2, std_vals2, min_vals2, max_vals2 = RR_utils.split_data(df, splits[fold], cfg.class_label, False, False) 

        if te_df.empty:
            l = [(tr_df, 'train', tr_df2)]
        else:
            l = [(tr_df, 'train', tr_df2), (te_df, 'test', te_df2)]

        for dataset in l:
            print('### {}:'.format(dataset[1]))
            X, Y = RR_utils.get_XY(dataset[0], cfg.task, cfg.class_label)
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100), link="logit")
            shap_values = explainer.shap_values(X, nsamples=10)
            for sv in range(len(shap_values)):
                array_sv = np.array(shap_values[sv])
                print(array_sv)
                print(array_sv.shape)
                columns = dataset[0].columns.tolist()
                columns.remove(cfg.class_label)
                sv_data = pd.DataFrame(array_sv, index=dataset[0].index.values.tolist(), columns=columns)
                sv_data.to_csv('{}{}_{}_{}_shap_values.csv'.format(out_fold+'shap_eval/', fold+1, dataset[1], sv))
                fp = shap.force_plot(explainer.expected_value[0], shap_values[sv], dataset[0][columns], link="logit", show=False)
                fig_file = '{}{}_{}_{}_force_plot.html'.format(out_fold+'shap_eval/', fold+1, dataset[1], sv)
                shap.save_html(fig_file, fp)
                #fp.savefig('{}{}_{}_{}_force_plot.pdf'.format(out_fold+'shap_eval/', fold+1, dataset[1], sv), bbox_inches='tight', )