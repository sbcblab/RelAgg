# Bruno Iochins Grisci
# May 24th, 2020

import matplotlib
matplotlib.use('Agg')
import os
import sys
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def wgt_cosine_distance(A, B):
    W = A[int(len(A)/2):]
    A = A[0:int(len(A)/2)]
    B = B[0:int(len(B)/2)]
    return 1.0 - (((A/np.linalg.norm(A))*(B/np.linalg.norm(B)))*(1.0-W)).sum()

def plot(df, features=None, norm=True, rescale=False, class_label='y', colors=list(colhex.keys()), markers=['o', 's', 'v', 'P', 'X', 'd', '*', 'h'], file_name='pca.png', method='tsne', task='classification', weights=None, perplexity=50, n_iter=5000):
    # Visualization

    marker_size = 50
    max_sample_size = 2000
    if len(df.index) > max_sample_size:
        print('\nWARNING: The dataset has over {} samples. Only {} random samples will be plotted!\n'.format(max_sample_size, max_sample_size))
        df = df.sample(n=max_sample_size, random_state=1)
        marker_size = 25

    if features is None:
        features = list(df.columns.values)
        features.remove(class_label)
    x = df.loc[:, features].values
    y = df.loc[:,[class_label]].values # Separating out the target
    if weights is not None:
        weights = weights.loc[features].values
    if norm:
        x = StandardScaler().fit_transform(x) # Standardizing the features
    if rescale:
        minVals = x.min(0)
        maxVals = x.max(0)
        max_min = maxVals - minVals
        max_min[max_min == 0] = 1
        x = (x - minVals) / max_min

    #print(x.shape)

    if method == 'pca':
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        xlab = 'PC 1 ({})'.format(round(pca.explained_variance_ratio_[0], 4))
        ylab = 'PC 2 ({})'.format(round(pca.explained_variance_ratio_[1], 4))
    elif method == 'tsne':
        if len(features) > 50 and weights is None:
            pca = PCA(n_components=50)
            principalComponents = pca.fit_transform(x)
            principalComponents = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter).fit_transform(principalComponents)
        elif weights is None:
            principalComponents = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter).fit_transform(x)
        else:
            W = np.tile(weights, (x.shape[0],1))
            b = np.concatenate((x, W), axis=1)
            principalComponents = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, metric=fast_wdist).fit_transform(b)
            #principalComponents = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, metric=wgt_cosine_distance).fit_transform(b)
        xlab = 'Component 1'
        ylab = 'Component 2'
    else:
        raise Exception('Unknown visualization method: {}'.format(method))

    principalDf = pd.DataFrame(data = principalComponents, index=df.index.values, columns = [xlab, ylab])
    finalDf = pd.concat([principalDf, df[[class_label]]], axis = 1)
    #print(finalDf)
    #print(pca.explained_variance_ratio_)
    finalDf.to_csv(file_name.replace('.png', '.csv').replace('.pdf', '.csv'))

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(xlab, fontsize = 15)
    ax.set_ylabel(ylab, fontsize = 15)
    ax.set_title(file_name, fontsize = 15)
    if task == 'classification':
        targets = np.sort(df[class_label].unique())
        t = 0
        for target in targets:
            indicesToKeep = finalDf[class_label] == target
            ax.scatter(finalDf.loc[indicesToKeep, xlab]
                       , finalDf.loc[indicesToKeep, ylab]
                       , c = colhex[colors[t%len(colors)]]
                       , marker = markers[t%len(markers)]
                       , s = marker_size)
            t = t+1
        ax.legend(targets)
    elif task == 'regression':
        target_color = finalDf[class_label].astype(float)
        s = ax.scatter(finalDf[xlab], finalDf[ylab], marker=markers[0], s=marker_size, c=target_color, cmap='coolwarm')
        cbar = plt.colorbar(s, ax=ax)
        cbar.set_label(class_label)
    else:
        raise Exception('Unknown task type: {}'.format(task))
    ax.grid()
    plt.savefig(file_name)
    plt.cla()
    plt.close()