dataset_file   = "DATA/XOR/xor_2in50_500.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
task           = "classification" # "classification" or "regression"
class_label    = "y"              # label of the column with the classes or target values
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS'        # name of directory in which the results will be saved

standardized        = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled            = False # True if data should be scaled between 0 and 1
train_epochs        = 150   # training epochs for training the neural networks
batch_size          = 4     # batch size for training the neural networks
weights_constraints = True  # True if neural network biases should be <= 0.0 and if weights and biases in the output layer must be >= 0.0 
k                   = 10    # number of folds for stratified k-fold cross-validation, if k <= 1 there is no partition and will use all samples
cv_splits           = None  # if None, the folds of stratified k-fold cross-validation will be divided randomly, if file path to split.py it will use split in the file
n_selection         = 2     # number of top ranked input features to be considered for further analyses

rel_rule       = ['alphabeta', 2.0] # the rule for computing the relevance, should be one of the below:
                                    # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww'], ['deeplift', reference]
                                    # epsilon and alpha must be integer or float
                                    # reference must be 'mean', 'zero', or float
rank           = "norm"       # 'rank' or 'norm', use 'norm' for the description in the publication
mean_type      = "geometric"  # 'geometric' or 'arithmetic', use 'geometric' for the description in the publication
rel_class      = "pred"       # 'pred', 'true', or class index (integer value), use "pred" for the description in the publication
eval_metric    = "f1score"    # 'f1score' or 'accuracy' (if task = "regression" it is automatically the MSE)
agglutinate    = False        # if True, the relevance score of a categorical input feature will be the average score of the individual binary categories
kendall_tau    = False        # if True, the kendall tau difference between the ranking scores of each fold will be computed

class_colors   = ['GREEN', 'ORANGE'] # list of colors to be assigned to the classes in the data, options as below:
                                     # 'RED', 'BLUE', 'YELLOW', 'GREEN', 'ORANGE', 'BLACK', 'CYAN', 'SILVER', 'MAGENTA', 'CREAM', 'DRKBRW', 'BEIGE', 'WHITE'
viz_method     = 'tsne' # 'pca' or 'tsne', use 'tsne' for the weighted t-SNE

# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
# Available activation functions are 'relu'
layers = [['dense', 20, 'relu'],
          ['dropout', 0.5],  
          ['dense', 10, 'relu'],
          ['dropout', 0.5],                       
         ]