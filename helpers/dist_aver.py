# Bruno Iochins Grisci
# May 24th, 2020

import os
import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':

    data_file = sys.argv[1]
    k = int(sys.argv[2])
    tt = int(sys.argv[3])
    df = pd.read_csv(data_file, delimiter=',', header=0, index_col=0)

    if tt > 0:
        train_test = []
        for fold in range(1,k+1):
            train_test.append(df['{}_train'.format(fold)]['{}_test'.format(fold)])
        print(train_test)
        print(len(train_test))
        train_test = np.array(train_test)
        print(train_test.mean(), train_test.std())

        print('###')

    data = df.values
    fold_d = []
    for i in range(len(data)):
        for j in range(i+1, len(data)): 
            fold_d.append(data[j][i])

    print(fold_d)
    print(len(fold_d))
    fold_d = np.array(fold_d)
    print(fold_d.mean(), fold_d.std())        
