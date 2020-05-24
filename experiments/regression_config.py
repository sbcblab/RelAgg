dataset_file   = "DATA/syn/regression_4in100_1000.csv"
task           = "regression"
target_split   = []
class_label    = "y"
train_epochs   = 150
batch_size     = 32
k              = 10
n_selection    = 4
cv_splits      = None
standardized   = True
rescaled       = False
rel_rule       = ['alphabeta', 2.0] # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww']
agglutinate    = False
kendall_tau    = True
rank           = "norm"       # 'rank' or 'norm'
mean_type      = "geometric"  # 'geometric' or 'arithmetic'
rel_class      = "pred"       # 'pred' 'true' or class index
eval_metric    = "f1score"    # 'f1score' or 'accuracy'
dataset_sep    = ","
output_folder  = 'RESULTS'
class_colors   = ['GREEN']
viz_method     = 'tsne' # 'pca' or 'tsne'
#class_colors   = ['CYAN', 'BEIGE', 'DRKBRW']
weights_constraints = True
# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
layers = [['dense', 256, 'relu'],
          ['dropout', 0.1],  
          ['dense', 256, 'relu'],
          ['dropout', 0.1],  
          ['dense', 128, 'relu'],                                           
         ]
