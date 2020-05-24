# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets

import plot_pca

def call_pca(df, features, file_name):
    # Visualization
    plot_pca.plot(df, norm=False, rescale=False, features=features, class_label='y', file_name=file_name, method='tsne', task='regression')

#####################################################################################################################

if __name__ == '__main__': 


    n_samples     = 1000
    n_features    = 100
    n_informative = 4
    n_targets     = 1 

    file_name = 'DATA/syn/{}_{}in{}_{}.csv'.format('regression', n_informative, n_features, n_samples)

    out_fold = file_name.replace(os.path.basename(file_name), '')
    if out_fold != '' and not os.path.exists(out_fold):
        os.makedirs(out_fold)

    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression
    #x, y, c = sklearn.datasets.make_regression(n_samples, n_features, n_informative, n_targets, coef=True, bias=500.0)
    #print(c)
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html#sklearn.datasets.make_sparse_uncorrelated
    x, y = datasets.make_sparse_uncorrelated(n_samples=n_samples, n_features=n_features, random_state=None)

    columns_labels = ["REL"+f'{i+1:03}' for i in range(0, n_informative)] + ["IRR"+f'{i+1:03}' for i in range(n_informative, n_features)]
    samples_labels = ["s"+f'{i+1:04}' for i in range(0, n_samples)]
    df = pd.DataFrame(x, index=samples_labels, columns =columns_labels) 
    df["y"] = y

    print(df) 

    print(y.mean(), y.std(), np.median(y), y.min(), y.max())

    df.to_csv(file_name)

    if n_informative > 0:
        call_pca(df, [f for f in columns_labels if "REL" in f], file_name.replace('.csv', '_REL.png'))
    if n_features - n_informative > 0:
        call_pca(df, [f for f in columns_labels if "IRR" in f], file_name.replace('.csv', '_IRR.png'))
    call_pca(df, columns_labels, file_name.replace('.csv', '_ALL.png'))