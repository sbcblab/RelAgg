import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main():
    '''
    file_name = '../weighted_tSNE/DATA/datasets/mnist_test.csv'
    df = pd.read_csv(file_name, delimiter=',', header=0, index_col=None)
    image = df.iloc[5]
    l = image['label']
    print('Label', l)
    image = image.drop('label')
    '''

    sel_file = 'RESULTS/wtsne/mnist_test/RelAgg_9_mnist_test.csv'
    image = pd.read_csv(sel_file, delimiter=',', header=0, index_col=0)

    #dummy = np.full((28, 28), 254)
    #print(dummy)

    image = image.values
    image = image.reshape(28, 28)
    #image = image * dummy

    rotated_image = image[::-1,] 
    print(rotated_image)
    print(rotated_image.shape)
    #print('Label', l)
    plt.imshow(rotated_image, 'gray', origin='lower')
    plt.show()

if __name__ == '__main__': 
    main()