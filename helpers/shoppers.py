# Bruno Iochins Grisci
# May 24th, 2020

import pandas as pd
import numpy as np

import plot_pca


if __name__ == '__main__': 
    df = pd.read_csv('DATA/shoppers/online_shoppers_intention.csv', delimiter=',', header=0)
    df.columns = df.columns.str.strip()

    print(df)
    print(df.columns)
    print(df['Revenue'].value_counts())

    df['Weekend'] = df['Weekend']*1
    print(df['VisitorType'].value_counts())
    print(df['Browser'].value_counts())
    print(df['Region'].value_counts())
    print(df['TrafficType'].value_counts())

    print(df)
    
    for feature in ['VisitorType', 'Month', 'Browser', 'Region', 'TrafficType']:
        categories = np.sort(df[feature].unique())
        for category in categories:
            df['{}_{}'.format(feature, category)] = (df[feature] == category)*1
        df = df.drop([feature], axis=1)

    df = df[ ['Revenue'] + [ col for col in df.columns if col != 'Revenue' ] ]

    print(df)
    print(df['Revenue'].value_counts())

    df.to_csv('DATA/shoppers/shoppers.csv')
    plot_pca.plot(df, norm=False, rescale='True', class_label='Revenue', colors=['ORANGE', 'GREEN'], file_name='DATA/shoppers/shoppers_plot.png', method='tsne')
