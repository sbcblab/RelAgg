# Bruno Iochins Grisci
# June 10th, 2022

dataset_file   = "DATA/enem/enem2016.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
task           = "regression" # "classification" or "regression"
class_label    = "NU_NOTA_MT"              # label of the column with the classes or target values
target_split   = [700]
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/movie'        # name of directory in which the results will be saved
row_index      = 0                # The column that has the row index, None if no index

standardized        = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled            = True # True if data should be scaled between 0 and 1
train_epochs        = 1000   # training epochs for training the neural networks
batch_size          = 64     # batch size for training the neural networks
weights_constraints = True  # True if neural network biases should be <= 0.0 and if weights and biases in the output layer must be >= 0.0 
k                   = 1     # number of folds for stratified k-fold cross-validation, if k <= 1 there is no partition and will use all samples
cv_splits           = None  # if None, the folds of stratified k-fold cross-validation will be divided randomly, if file path to split.py it will use split in the file
checkpoint          = 1000   # periodicity to save the models, if it should not be used set to be equal to train_epochs. If equals to 0 or 1 will save all epochs.
n_selection         = 2     # number of top ranked input features to be considered for further analyses

regularizer         = None  # weights regularizer: 'l1', 'l2', or None
regularization      = 0.0  # parameter for the regularizer, 0.0 if not used

rel_rule       = ['alphabeta', 2.0] # the rule for computing the relevance, should be one of the below:
                                    # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww'], ['deeplift', reference]
                                    # epsilon and alpha must be integer or float
                                    # reference must be 'mean', 'zero', or float
rank           = "norm"       # 'rank' or 'norm', use 'norm' for the description in the publication
mean_type      = "geometric"  # 'geometric' or 'arithmetic', use 'geometric' for the description in the publication
rel_class      = "pred"       # 'pred', 'true', or class index (integer value), use "pred" for the description in the publication
eval_metric    = "MSE"    # 'f1score' or 'accuracy' (if task = "regression" it is automatically the MSE)
agglutinate    = False        # if True, the relevance score of a categorical input feature will be the average score of the individual binary categories
kendall_tau    = False        # if True, the kendall tau difference between the ranking scores of each fold and between training and test sets will be computed

class_colors   = ['GREEN', 'ORANGE'] # list of colors to be assigned to the classes in the data, options as below:
                                     # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
viz_method     = 'tsne' # 'pca' or 'tsne', use 'tsne' for the weighted t-SNE
perplexity     = 50     # perplexity value for t-SNE
n_iter         = 5000   # number of iterations for t-SNE

# List of neural networks layers for creation of keras sequential model,
# except the input and output layers that are created based on the number of input features, number of classes, and type of task. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
# Available activation functions are 'relu'
layers = [['dense', 256, 'relu'],
          ['dense', 256, 'relu'],      
          ['dense', 128, 'relu'],                            
         ]

# NOTES:
#
# 1. If you don't want to use data partitions, set k = 1.
# 2. If you want to use your own data partitions, create a file called 'split.py' and put it in the directory specified in output_folder. 
#    Below is an example of split.py file with 6 samples divided in 3 folds:
#
'''
from collections import namedtuple
from numpy import array
Split = namedtuple('Split', ['tr', 'te'])
splits = [Split(tr=array([  0,   2,   3,   4]), 
                te=array([  1,   5,   6])), 
          Split(tr=array([  1,   4,   5,   6]), 
                te=array([  0,   2,   3])), 
          Split(tr=array([  0,   1,   2,   3]), 
                te=array([  4,   5,   6]))]
'''
#
# 3. If you want to use relevance aggregation on your own Keras models, follow these steps:
#    - Train a model using Keras with only dense relu or dropout layers, and linear or softmax layers for the output.
#    - Save your model in the .hdf5 file format.
#    - Rename the saved model file to 'x_1.hdf5', in which x = the file name of the dataset as specified in dataset_file.
#    - Delete the 'x_1.hdf5' present in the directory specified in output_folder and put the file you just created in its place.
#    - Run python3 get_relevances.py config.py as usual, now the scripts will use your own model.
#
# 4. For more information about weights_constraints, read Section 3.2 of the main paper.
# 5. For more information about agglutinate, read Section 3.1 of the main paper.
# 6. for more information about weighted t-SNE and table heatmaps, read Section 3.4 of the main paper.