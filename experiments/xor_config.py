dataset_file   = "DATA/XOR/xor_2in50_500.csv"
task           = "classification"
class_label    = "y"
train_epochs   = 150
batch_size     = 4
k              = 10
n_selection    = 2
cv_splits      = None
standardized   = False
rescaled       = False
rel_rule       = ['alphabeta', 2.0] # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww']
agglutinate    = False
kendall_tau    = True
rank           = "norm"      # 'rank' or 'norm'
mean_type      = "geometric" # 'geometric' or 'arithmetic'
rel_class      = "pred"      # 'pred' 'true' or class index
eval_metric    = "f1score"   # 'f1score' or 'accuracy'
dataset_sep    = ","
output_folder  = 'RESULTS'
class_colors   = ['GREEN', 'ORANGE']
viz_method     = 'tsne' # 'pca' or 'tsne'
#class_colors   = ['CYAN', 'BEIGE', 'DRKBRW']
# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
weights_constraints = True
layers = [['dense', 20, 'relu'],
          ['dropout', 0.5],  
          ['dense', 10, 'relu'],
          ['dropout', 0.5],                       
         ]
