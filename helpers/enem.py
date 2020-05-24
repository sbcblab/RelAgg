# Bruno Iochins Grisci
# May 24th, 2020

import pandas as pd
import numpy as np

import plot_pca


if __name__ == '__main__': 
    
    LABEL = 'NU_NOTA_MT'

    df = pd.read_csv('DATA/enem/enem2.csv', delimiter=',', header=0)
    df.columns = df.columns.str.strip()

    print(df)
    print(df.columns)
    for c in df.columns:
        print(c)
    df = df.dropna(how='all') 
    df = df.dropna(subset=['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'TP_LINGUA', 'NU_NOTA_REDACAO']) 

    for test in ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']:
        df = df[df[test] > 0]
        df = df[df[test] <= 1000]

    df['SG_UF_NASCIMENTO'] = df['SG_UF_NASCIMENTO'].fillna('outside')
    df['TP_ESTADO_CIVIL'] = df['TP_ESTADO_CIVIL'].fillna(0)
    df['TP_ESTADO_CIVIL'] = df['TP_ESTADO_CIVIL'].map({0:'single', 1:'married', 2:'divorced', 3:'widow'})
    df['TP_SEXO'] = df['TP_SEXO'].map({'M':0, 'F':1})
    
    df['TP_COR_RACA'] = df['TP_COR_RACA'].map({0:'not_informed', 1:'white', 2:'black', 3:'brown', 4:'yellow', 5:'native', 6:'not_informed'})
    df['TP_NACIONALIDADE'] = df['TP_NACIONALIDADE'].map({0:'not_informed', 1:'brazilian', 2:'naturalized', 3:'foreigner', 4:'born_abroad'})
    df['TP_ST_CONCLUSAO'] = df['TP_ST_CONCLUSAO'].map({1:'concluded', 2:'in2016', 3:'after2016', 4:'not_enrolled'})
    df['TP_ESCOLA'] = df['TP_ESCOLA'].map({1:'not_informed', 2:'public', 3:'private', 4:'abroad'})

    df['Q027'] = df['Q027'].fillna(26)
    df['Q027'] = df['Q027'].map({26:26, 'A': 0, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25}) 

    df['Q028'] = df['Q028'].fillna(0)
    df['Q028'] = df['Q028'].map({0: 0, 'A': 1, 'B': 11, 'C': 21, 'D': 31, 'E': 41}) 

    for col in ['Q029', 'Q030', 'Q031', 'Q032', 'Q033', 'Q034','Q035','Q036','Q037','Q038','Q039', 'Q040', 'Q041']:
        df[col] = df[col].fillna(0)

    df['Q001'] = df['Q001'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G':7, 'H':0})
    df['Q002'] = df['Q002'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G':7, 'H':0})

    df['Q006'] = df['Q006'].map({'A': 0.0, 'B': 880.0, 'C': 1320.0, 'D': 1760.0, 'E':2200.0, 'F': 2640.0, 'G':3520.0, 'H': 4400.0, 'I':5280.0, 'J':6160.0, 'K':7040.0, 
                                 'L': 7920.0, 'M': 8800.0, 'N': 10560.0, 'O':13200.0, 'P':17600.0, 'Q':17601.0})

    df['Q007'] = df['Q007'].map({'A':0, 'B':1, 'C':3, 'D':5})
    for col in ['Q008', 'Q009', 'Q010', 'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025']:
        df[col] = df[col].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4})
    
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)

    #for col in ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']:
    #    df.drop(col, inplace=True, axis=1)

    categorical_features = ['SG_UF_RESIDENCIA', 'TP_ESTADO_CIVIL', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'SG_UF_NASCIMENTO', 'TP_ST_CONCLUSAO', 'TP_ESCOLA', 'SG_UF_PROVA', 'Q003', 'Q004','Q026', 'Q042','Q043' ,'Q044','Q045','Q046','Q047', 'Q048','Q049','Q050']
    for feature in categorical_features:
        categories = np.sort(df[feature].unique())
        for category in categories:
            df['{}.{}'.format(feature, category)] = (df[feature] == category)*1
        df = df.drop([feature], axis=1)

    df = df[ [LABEL] + [ col for col in df.columns if col != LABEL ] ]

    print(df)
    print(df[LABEL].value_counts())
    print(df[LABEL].mean(), df[LABEL].std(), df[LABEL].median(), df[LABEL].min(), df[LABEL].max())
    print(df.columns[df.isnull().any()].tolist())

    df.to_csv('DATA/enem/enem2016.csv')
    plot_pca.plot(df, norm=True, rescale=False, class_label=LABEL, colors=['GREEN'], file_name='DATA/enem/enem2016_plot.pdf', method='tsne', task='regression')
