# Bruno Iochins Grisci
# October 15th, 2020

import os
import sys
import pandas as pd
import numpy as np

if __name__ == '__main__': 

    ranking_file = sys.argv[1]
    gpl_file     = sys.argv[2]

    df_rank = pd.read_csv(ranking_file, delimiter=',', header=0, index_col=0)
    df_gpl  = pd.read_csv(gpl_file, delimiter='\t', header=0, index_col=0)
    print(df_rank)
    print(df_gpl)

    #df_conv = df_gpl[['ProbeName', 'GB_ACC', 'GeneName']]
    #df_conv.set_index('ProbeName', inplace=True)
    df_conv = df_gpl[['Gene Symbol']]
    #df_conv.set_index('ID', inplace=True)
    print(df_conv)

    #print('NM_020812' in df_conv.index)
    #print(df_conv.at['NM_020812', 'GeneName'])

    new_indexes = {}
    for i in df_rank.index:
        #name = set(list(df_conv[df_conv['ProbeName'] == i]['GeneName'].values) + list(df_conv[df_conv['GB_ACC']    == i]['GeneName'].values))
        if i in df_conv.index:
            #print(df_conv.loc[i, 'Gene Symbol'])
            name = df_conv.loc[i, 'Gene Symbol']
        else:
            name = i
        if pd.isnull(name) or name == '---':
            name = i
        new_indexes[i] = name
        #if len(name) != 1:
        #    print(i, name)
        #print(i, name)
        #print(df_conv[df_conv['ProbeName'] == i]['GeneName'].values)
        #print(df_conv[df_conv['GB_ACC']    == i]['GeneName'].values)
        #print(df_conv[df_conv['GeneName']  == i]['GeneName'].values)
        #print('...')

    df = df_rank.rename(index = new_indexes)   
    #df = df_rank.rename(index = lambda s : s if (s not in df_conv['ProbeName'].values) else df_conv.at[s, 'GeneName'])
    #df = df_rank.rename(index = lambda s : s if (s not in df_conv['GB_ACC'].values) else df_conv.at[s, 'GeneName'])
    print(df)
    df.to_csv(ranking_file.replace('.csv', '_GPL.csv'))