dataset_file   = "DATA/Breast_GSE45827.csv"
task           = "classification"
class_label    = "type"
train_epochs   = 100
batch_size     = 32
k              = 3
n_selection    = 10
cv_splits      = None
standardized   = True
rescaled       = False
rel_rule       = ['deeplift', 0.0] # ['simple'], ['epsilon', epsilon], ['alphabeta', alpha], ['flat'], ['ww']
agglutinate    = False
kendall_tau    = True
rank           = "norm"      # 'rank' or 'norm'
mean_type      = "geometric" # 'geometric' or 'arithmetic'
rel_class      = "pred"      # 'pred' 'true' or class index
eval_metric    = "f1score"   # 'f1score' or 'accuracy'
dataset_sep    = ","
output_folder  = 'RESULTS'
class_colors   = ['CYAN', 'GREEN', 'YELLOW', 'BEIGE', 'DRKBRW', 'BLACK', 'RED']
viz_method     = 'tsne' # 'pca' or 'tsne'
perplexity     = 50     # perplexity value for t-SNE
n_iter         = 5000   # number of iterations for t-SNE
#class_colors   = ['CYAN', 'BEIGE', 'DRKBRW']
# List of neural networks layers for creation of keras sequential model,
# except the output layer that is created based on the number of classes and 'use softmax'. 
# The inputs are automatically set from the number of features in the dataset.
# Available layers types are dense and dropout.
weights_constraints = True
layers = [
          ['dense', 100, 'relu'],
          ['dropout', 0.5],
          ['dense', 200, 'relu'],
          ['dropout', 0.5],           
          ['dense', 50, 'relu'],                    
         ]
