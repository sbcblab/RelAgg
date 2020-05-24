# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import pandas as pd
import sklearn
from sklearn import datasets

import plot_pca

def call_pca(df, features, file_name):
    # Visualization
    plot_pca.plot(df, features=features, class_label='y', file_name=file_name, method='tsne')

#####################################################################################################################

if __name__ == '__main__': 
    n_samples = 1000
    n_classes = 3
    class_sep = 1.0

    n_relevant = 3
    n_redundant = 2
    n_repeated = 0
    n_irrelevant = 995
    n_features = n_relevant + n_redundant + n_repeated + n_irrelevant

    file_name = 'DATA/syn/{}_{}in{}_{}.csv'.format(n_classes, n_relevant+n_redundant, n_features, n_samples)

    out_fold = file_name.replace(os.path.basename(file_name), '')
    if out_fold != '' and not os.path.exists(out_fold):
        os.makedirs(out_fold)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
    x, y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_relevant, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=False, random_state=None)

    columns_labels = ["REL"+f'{i+1:03}' for i in range(0, n_relevant)] + ["RED"+f'{i+1:03}' for i in range(n_relevant, n_relevant+n_redundant)] + ["REP"+f'{i+1:03}' for i in range(n_relevant+n_redundant, n_relevant+n_redundant+n_repeated)] + ["IRR"+f'{i+1:03}' for i in range(n_relevant+n_redundant+n_repeated, n_relevant+n_redundant+n_repeated+n_irrelevant)]
    samples_labels = ["s"+f'{i+1:04}'+"_"+str(y[i]) for i in range(0, n_samples)]
    df = pd.DataFrame(x, index=samples_labels, columns =columns_labels) 
    df["y"] = y

    print(df) 

    df.to_csv(file_name)

    if n_relevant > 0:
        call_pca(df, [f for f in columns_labels if "REL" in f], file_name.replace('.csv', '_REL.png'))
    if n_redundant > 0:
        call_pca(df, [f for f in columns_labels if "RED" in f], file_name.replace('.csv', '_RED.png'))
    if n_repeated > 0:
        call_pca(df, [f for f in columns_labels if "REP" in f], file_name.replace('.csv', '_REP.png'))
    if n_irrelevant > 0:
        call_pca(df, [f for f in columns_labels if "IRR" in f], file_name.replace('.csv', '_IRR.png'))
    if n_relevant + n_redundant > 0:
        call_pca(df, [f for f in columns_labels if "REL" in f or "RED" in f], file_name.replace('.csv', '_RELRED.png'))
    call_pca(df, columns_labels, file_name.replace('.csv', '_ALL.png'))