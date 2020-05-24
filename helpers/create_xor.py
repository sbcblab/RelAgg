# Bruno Iochins Grisci
# May 24th, 2020

import numpy as np
import pandas as pd

import plot_pca

def call_pca(df, features, file_name):
    # Visualization
    plot_pca.plot(df, features=features, norm=False, class_label='y', file_name=file_name)

def xor(x):
    return 1 if np.array(x).sum() == 1 else 0

if __name__ == '__main__': 
    
    n_samples    = 500
    n_relevant   = 2
    n_irrelevant = 50 - n_relevant

    x = []
    y = []

    h = int(n_samples/2)
    for i in range(0, h):
        label = -1
        while label != 0:
            feats = np.random.choice([0, 1], size=(n_relevant+n_irrelevant,), p=[1./2, 1./2])
            label = xor(feats[0:n_relevant])
            print(label)
        x.append(feats)
        y.append(0)
        print(i)

    print('Zero ok')

    for i in range(h, n_samples):
        label = -1
        while label != 1:
            feats = np.random.choice([0, 1], size=(n_relevant+n_irrelevant,), p=[1./2, 1./2])
            label = xor(feats[0:n_relevant])
            print(label)
        x.append(feats)
        y.append(1)
        print(i)

    print('One ok')

    columns_labels = ["REL"+f'{i+1:03}' for i in range(0, n_relevant)] + ["IRR"+f'{i+1:03}' for i in range(n_relevant, n_relevant+n_irrelevant)]
    print(columns_labels)
    samples_labels = ["s"+f'{i+1:04}'+"_"+str(y[i]) for i in range(0, n_samples)]
    print(samples_labels)
    df = pd.DataFrame(x, index=samples_labels, columns =columns_labels) 
    #df["y"] = y
    df.insert(0, 'y', y, True) 

    print(df) 

    file_name = 'DATA/XOR/regxor_{}in{}_{}.csv'.format(n_relevant, n_relevant+n_irrelevant, n_samples)
    df.to_csv(file_name)

    if n_relevant > 0:
        call_pca(df, [f for f in columns_labels if "REL" in f], file_name.replace('.csv', '_REL.png'))
    if n_irrelevant > 0:
        call_pca(df, [f for f in columns_labels if "IRR" in f], file_name.replace('.csv', '_IRR.png'))
    #call_pca(df, columns_labels, file_name.replace('.csv', '_ALL.png'))