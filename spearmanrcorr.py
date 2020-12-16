# Bruno Iochins Grisci
# December 15th, 2020

import os
import sys
import importlib
import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from numpy.random import rand
from numpy.random import seed
seed(42)


regs = ['RESULTSRAND/1REG/3_5in1000_1000_1_train_relsheet.csv', 'RESULTSRAND/2REG/3_5in1000_1000_1_train_relsheet.csv', 'RESULTSRAND/3REG/3_5in1000_1000_1_train_relsheet.csv']
dats = ['RESULTSRAND/1DATARAND/3_5in1000_1000_rand_1_train_relsheet.csv', 'RESULTSRAND/2DATARAND/3_5in1000_1000_rand_1_train_relsheet.csv', 'RESULTSRAND/3DATARAND/3_5in1000_1000_rand_1_train_relsheet.csv']
weis = ['RESULTSRAND/1W/3_5in1000_1000_1_train_relsheet.csv', 'RESULTSRAND/2W/3_5in1000_1000_1_train_relsheet.csv', 'RESULTSRAND/3W/3_5in1000_1000_1_train_relsheet.csv']

coefs = []

for r in regs:
    for d in regs:
        if r != d:
            data1 = pd.read_csv(r, delimiter=',', header=0, index_col=0)
            data1.drop(index=['usage','prob_0','prob_1','prob_2','brute_0','brute_1','brute_2','max_out','prediction','y'], inplace=True)
            data1 = data1['score']
            data1 = data1.sort_index()
            print(data1)

            data2 = pd.read_csv(d, delimiter=',', header=0, index_col=0)
            data2.drop(index=['usage','prob_0','prob_1','prob_2','brute_0','brute_1','brute_2','max_out','prediction','y'], inplace=True)
            data2 = data2['score']
            data2 = data2.sort_index()
            print(data2)

            # calculate spearman's correlation
            coef, p = spearmanr(data1, data2)
            print('Spearmans correlation coefficient: %.3f' % coef)
            coefs.append(coef)
            # interpret the significance
            alpha = 0.05
            if p > alpha:
                print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
            else:
                print('Samples are correlated (reject H0) p=%.3f' % p)

coefs = np.array(coefs)
print(coefs.mean())

# regs x dats = -0.011490310156976826
# regs x weis = 0.006792917459584127
# regs x regs = 0.0358936118936119