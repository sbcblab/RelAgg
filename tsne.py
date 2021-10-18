# Bruno Iochins Grisci
# December 4th, 2020

import os
import sys
import importlib
import pandas as pd

from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utilstsne


import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

colhex = {
    'RED':     '#BA0000',
    'BLUE':    '#0000FF',
    'YELLOW':  '#FFEE00',
    'GREEN':   '#048200',    
    'ORANGE':  '#FF6103',
    'BLACK':   '#000000',
    'CYAN':    '#00FFD4',    
    'SILVER':  '#c0c0c0',
    'MAGENTA': '#680082',
    'CREAM':   '#FFFDD0',
    'DRKBRW':  '#654321',
    'BEIGE':   '#C2C237',
    'WHITE':   '#FFFFFF',
}

def fast_wdist(A, B):
    W = A[int(len(A)/2):]
    A = A[0:int(len(A)/2)]
    B = B[0:int(len(B)/2)]
    return ((W*(A-B))**2).sum()**(0.5)

if __name__ == '__main__': 

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    print(df)

    if cfg.standardized:
    #if False:
        df, meanVals, stdVals = RR_utils.standardize(df, cfg.class_label)

    x = df.drop(cfg.class_label, axis=1).to_numpy()
    y = df[cfg.class_label].astype(str)

    print("Data set contains %d samples with %d features" % x.shape)

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)

    #print("%d training samples" % x_train.shape[0])
    #print("%d test samples" % x_test.shape[0])

    rel = pd.read_csv("RESULTS/3_5in1000_1000/3_5in1000_1000_1_train_relevance.csv", delimiter=cfg.dataset_sep, header=0, index_col=0)
    print(rel)
    #rel = rel.abs().div(rel.abs().max(axis=1), axis=0)
    print(rel)
    W = rel.to_numpy()
    #wx = W * x 
    wx = np.multiply(W,x)

    print("-----")
    print(df)
    print(rel)

    '''

    weights = np.zeros(x.shape[1])
    #weights = np.zeros(x.shape[1])
    weights[0] = 1
    weights[1] = 1
    weights[2] = 1
    weights[3] = 1
    weights[4] = 1
    #W = np.tile(weights, (x.shape[0],1))
    #wx = np.concatenate((x, W), axis=1)
    wx = weights * x

    print(weights)
    print(weights.shape)
    #print(W)
    #print(W.shape)
        '''
    print(wx)
    print(wx.shape)

    tsne = TSNE(
        perplexity=max(30, x.shape[0]/100),
        initialization="pca",
        #metric=fast_wdist,
        metric='euclidean',
        neighbors="auto",
        learning_rate="auto",
        verbose=True,
        n_jobs=8,
        n_iter=500,
        random_state=42,
    )

    embedding = tsne.fit(W)

    print(embedding)

    emb = pd.DataFrame(data=embedding,    # values
                       index=df.index,    # 1st column as index
                       columns=['tsne1', 'tsne2'])  # 1st row as the column names
    emb[cfg.class_label] = y
    emb.to_csv(os.path.basename(cfg.dataset_file).replace('.csv','_tsne.csv'))

    print(emb)

    COL = {}
    CLASS_LABELS = list(np.sort(df[cfg.class_label].astype(str).unique()))
    for c, l in zip(cfg.class_colors, CLASS_LABELS):
        COL[l] = colhex[c]
    print(COL)

    ax = utilstsne.plot(embedding, y, colors=COL, title=os.path.basename(cfg.dataset_file).replace('.csv',''), draw_centers=True, draw_cluster_labels=False, s=10)
    plt.savefig(os.path.basename(cfg.dataset_file).replace('.csv','_force_plot.pdf'), bbox_inches='tight')

