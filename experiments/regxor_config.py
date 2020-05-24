dataset_file   = "DATA/XOR/regxor_2in50_500.csv"
task           = "regression"
class_label    = "y"
target_split   = [0.5]
train_epochs   = 150
batch_size     = 4
k              = 10
n_selection    = 2
cv_splits      = None
standardized   = False
rescaled       = False
rel_rule       = ['alphabeta', 2.0] # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww'], ['deeplift', reference]
agglutinate    = False
kendall_tau    = True
rank           = "norm"      # 'rank' or 'norm'
mean_type      = "geometric" # 'geometric' or 'arithmetic'
rel_class      = "pred"      # 'pred' 'true' or class index
eval_metric    = "f1score"   # 'f1score' or 'accuracy'
dataset_sep    = ","
output_folder  = 'RESULTS'
class_colors   = ['GREEN']
viz_method     = 'tsne' # 'pca' or 'tsne'

weights_constraints = True
# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
layers = [['dense', 20, 'relu'],
          ['dropout', 0.1],  
          ['dense', 10, 'relu'],                       
         ]
