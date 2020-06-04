dataset_file   = "DATA/enem/enem2016.csv"
task           = "regression"
target_split   = [700]
class_label    = "NU_NOTA_MT"
train_epochs   = 200
batch_size     = 64
k              = 10
n_selection    = 5
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
perplexity     = 50     # perplexity value for t-SNE
n_iter         = 5000   # number of iterations for t-SNE
#class_colors   = ['CYAN', 'BEIGE', 'DRKBRW']
# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
weights_constraints = True
layers = [['dense', 256, 'relu'],
          ['dropout', 0.2],  
          ['dense', 256, 'relu'],    
          ['dropout', 0.2],  
          ['dense', 128, 'relu'],                            
         ]