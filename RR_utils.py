# Bruno Iochins Grisci
# May 24th, 2020

import os
import itertools
from collections import namedtuple
import tempfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, KFold
import scipy.stats as stats

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.utils import plot_model, to_categorical

######################################################################

class NonPos(Constraint):
    """Constrains the weights to be non-positive.
    """

    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.), K.floatx())

class IsZero(Constraint):
    """Constrains the weights to be zero.
    """

    def __call__(self, w):
        return w * K.cast(K.equal(w, 0.), K.floatx()) 

######################################################################

def create_output_dir(data_file, output_folder='', dataset_format='.csv'):
    if len(output_folder) > 1:
        if output_folder[-1] != '/':
            output_folder = output_folder + '/'
    out_fold  = output_folder + os.path.basename(data_file).replace(dataset_format, '/')
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    out_file = out_fold + os.path.basename(data_file).replace(dataset_format, '')
    return out_fold, out_file

def check_dataframe(df, class_label, task):
    df.columns = df.columns.str.strip()
    idx = df.index[df.index.duplicated()].unique()
    if len(idx) > 0:
        raise Exception('\nERROR: Repeated row indices in dataset: {}'.format(idx))
    idx = df.columns[df.columns.duplicated()].unique()
    if len(idx) > 0:
        raise Exception('\nERROR: Repeated column indices in dataset: {}'.format(idx))
    if class_label not in df.columns:
        raise Exception('\nERROR: Could not find class label "{}" in dataset columns: {}'.format(class_label, df.columns))
    if task == 'classification':
        df[class_label] = df[class_label].astype(str).str.strip()
    elif task == 'regression':
        df[class_label] = df[class_label].astype(float)
    return df

def min_max_scaling(df, class_label, minVals=None, maxVals=None):
    if df.empty:
        return df, None, None
    y = df[class_label].values
    y_index = list(df.columns.values).index(class_label)
    val_array = df.drop([class_label], axis=1).values
    if minVals is None:
        minVals = val_array.min(0)
    if maxVals is None:
        maxVals = val_array.max(0)
    max_min = maxVals - minVals
    max_min[max_min == 0] = 1
    val_array = (val_array - minVals) / max_min
    list_wo_y = list(df.columns.values)
    list_wo_y.remove(class_label)
    new_df = pd.DataFrame(val_array, index=list(df.index.values), columns=list_wo_y)
    new_df.insert(y_index, class_label, y, True) 
    return new_df, minVals, maxVals

def standardize(df, class_label, meanVals=None, stdVals=None):
    if df.empty:
        return df, None, None
    y = df[class_label].values
    y_index = list(df.columns.values).index(class_label)
    val_array = df.drop([class_label], axis=1).values
    if meanVals is None:
        meanVals = val_array.mean(0)
    if stdVals is None:
        stdVals = val_array.std(0)
    stdVals[stdVals == 0] = 1
    val_array = (val_array - meanVals) / (stdVals)
    list_wo_y = list(df.columns.values)
    list_wo_y.remove(class_label)
    new_df = pd.DataFrame(val_array, index=list(df.index.values), columns=list_wo_y)
    new_df.insert(y_index, class_label, y, True) 
    return new_df, meanVals, stdVals

def split_cv(df, task, class_label, k):
    X = df.drop([class_label],axis=1)
    y = df[class_label]
    if k > 1:
        if task == 'classification':
            skf = StratifiedKFold(n_splits=k, shuffle=True)
        elif task == 'regression':
            skf = KFold(n_splits=k, shuffle=True)
        skf.get_n_splits(X, y)
        splits = []
        for tri, tei in skf.split(X, y):
            Split = namedtuple('Split', ['tr', 'te'])
            splits.append(Split(tr=tri, te=tei))
    else:
        Split = namedtuple('Split', ['tr', 'te'])
        splits = [Split(tr=np.arange(0, len(X.index)), te=None)]
    return splits

def split_data(df, splits, class_label, standardized, rescaled):
    tr_df = df.iloc[splits.tr]
    if splits.te is not None:
        te_df = df.iloc[splits.te]
    else:
        te_df = pd.DataFrame()

    mean_vals = None
    std_vals  = None
    if standardized:
        tr_df, mean_vals, std_vals = standardize(tr_df, class_label=class_label)
        te_df, mean_vals, std_vals = standardize(te_df, class_label=class_label, meanVals=mean_vals, stdVals=std_vals)
    min_vals = None
    max_vals = None
    if rescaled:
        tr_df, min_vals, max_vals = min_max_scaling(tr_df, class_label=class_label)
        te_df, min_vals, max_vals = min_max_scaling(te_df, class_label=class_label, minVals=min_vals, maxVals=max_vals)

    return tr_df, te_df, mean_vals, std_vals, min_vals, max_vals    

def get_XY(df, task, class_label):
    if df.empty:
        return None, None
    if type(class_label) is not list:
        class_label = [class_label]
    cl = class_label[0]
    dd = df
    for c in class_label:
        dd = dd.drop([c], axis=1)
    X = dd.values
    if task == 'classification':
        Y = pd.get_dummies(df[cl]).values
    elif task == 'regression':
        Y = df[cl].values
    return X, Y

def write_model(model, file_name):
    with open(file_name, 'w') as f:
        for layer in model.layers:
            print('### LAYER')
            if 'activation' in layer.get_config():
                act = layer.get_config()['activation'].capitalize()
                if act == 'Relu':
                    act = 'Rect'
                elif act == 'Softmax':
                    act = 'Flatten' # Write softmax layer as relu for LRP
                elif act == 'Linear':
                    act = 'Rect'
                print(act)
                len_input, len_output = layer.input_shape[1], layer.output_shape[1]
                print(len_input, len_output)
                weights_bias = layer.get_weights()
                if len(weights_bias) > 0:
                    weights      = weights_bias[0]
                    bias         = weights_bias[1]
                    print('weights')
                    print('bias')
                    f.write('Linear {} {}\n'.format(len_input, len_output))
                    f.write(' '.join([repr(w) for w in weights.flatten()]) + '\n')
                    f.write(' '.join([repr(b) for b in bias.flatten()]) + '\n')
                    f.write('{}\n'.format(act))

def save_accuracy(tr_accuracy, te_accuracy, k, file_name):
    tr_accuracy = np.array(tr_accuracy)
    te_accuracy = np.array(te_accuracy)
    with open(file_name, 'w') as acc_file:
        acc_file.write('TRAINING\n')
        print('TRAINING')
        acc_file.write(str(tr_accuracy) + '\n')
        print(str(tr_accuracy))
        acc_file.write('{} | {}\n'.format(tr_accuracy.mean(), tr_accuracy.std()))      
        print('{} | {}\n'.format(tr_accuracy.mean(), tr_accuracy.std()))        
        if k > 1:
            acc_file.write('\nTESTING\n')
            print('TESTING')
            acc_file.write(str(te_accuracy) + '\n')
            print(str(te_accuracy))
            acc_file.write('{} | {}\n'.format(te_accuracy.mean(), te_accuracy.std()))
            print('{} | {}\n'.format(te_accuracy.mean(), te_accuracy.std()))

def save_usage(df, splits, file_name):
    df_usage = pd.DataFrame(np.zeros((len(df.index.values),1)), index=list(df.index.values), columns =["usage"])
    df_usage.iloc[splits.tr] = "train"
    df_usage.iloc[splits.te] = "test"
    df_usage.to_csv(file_name)    

def confusion_matrix(df, class_label, model, file_name):
    X, Y = get_XY(df, 'classification', class_label)
    classes = np.sort(df[class_label].unique())
    #y_pred = model.predict_classes(X)
    # https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
    y_pred1 = model.predict(X)
    y_pred  = np.argmax(y_pred1, axis=1)
    con_mat = tensorflow.math.confusion_matrix(labels=np.array([np.where(classes==c)[0][0] for c in df[class_label].values]), predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
    con_mat_df.to_csv(file_name)

def regression_distance(df, class_label, model, file_name):
    X, Y = get_XY(df, 'regression', class_label)
    estimate = model.predict(X)
    index = np.array([str(v) for v in df.index.values])
    comp = pd.DataFrame([], index=index, columns=['prediction', 'target'])
    comp['prediction'] = estimate
    comp['target'] = Y
    comp.sort_values(by='target', inplace=True)
    comp.to_csv(file_name)
    comp.plot.line(style=[':', '.']).get_figure().savefig(file_name.replace('.csv', '.pdf'))

def apply_modifications(model, custom_objects=None):

    # code from https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py

    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    finally:
        os.remove(model_path)

def save_outputs(df, splits, task, class_label, model, file_name, rescale, standard, minVals=None, maxVals=None, meanVals=None, stdVals=None):
    if rescale:
        df, mv, xv = min_max_scaling(df, class_label, minVals, maxVals)
    if standard:
        df, md, sd = standardize(df, class_label, meanVals, stdVals)

    df_out = pd.DataFrame(np.zeros((len(df.index.values),1)), index=list(df.index.values), columns =["usage"])
    df_out.iloc[splits.tr] = "train"
    if splits.te is not None:
        df_out.iloc[splits.te] = "test"
    
    X, Y = get_XY(df, task, class_label)
    pred = model.predict(X)
    if task == 'classification':
        classification = model.predict_classes(X)
        classes = np.sort(df[class_label].unique())
        
        model2 = Sequential()
        for layer in model.layers:
            model2.add(layer)
        model2.layers[-1].activation = activations.linear
        model2 = apply_modifications(model2, {'NonPos': NonPos, 'IsZero': IsZero})

        brute_out = model2.predict(X)
    elif task == 'regression':
        classification = np.zeros(pred.shape)
        classes = np.array(['target'])
        brute_out = pred

    for c in range(len(classes)):
        df_out["prob_{}".format(classes[c])] = pred[:,c]
    for o in range(len(brute_out[0])):
        df_out["brute_{}".format(classes[o])] = brute_out[:,o]
    df_out["max_out"] = brute_out.max(axis=1)
    df_out["prediction"] = [classes[int(c)] for c in classification]
    df_out[class_label] = df[class_label].values
    df_out.to_csv(file_name)

def get_class_weights(labels, values, index_type='label'):
    # https://androidkt.com/set-class-weight-for-imbalance-dataset-in-keras/
    class_weights_values = class_weight.compute_class_weight('balanced', labels, values)
    if index_type == 'label':
        return dict(zip(labels, class_weights_values))
    elif index_type == 'index':
        return dict(enumerate(class_weights_values))
    else:
        raise Exception('Unknown index type: {}'.format(index_type))   

def kendall_tau_distance(order_a, order_b):
    order_a = list(order_a)
    order_b = list(order_b)
    n = len(order_a)
    distance = 0
    pairs = itertools.combinations(list(set(order_a+order_b)), 2)
    for x, y in pairs:
        #print('x = {} oa[{}] ob[{}] y = {} oa[{}] ob[{}]'.format(x, order_a.index(x), order_b.index(x), y, order_a.index(y), order_b.index(y)))
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        #print('a = {} b = {}'.format(a, b))
        if a * b < 0:
            distance += 1
        #print('distance = {}'.format(distance))
    return distance / (n*(n-1)/2.0)

def kendall_tau(ranks, labels, n_selection):
    matrix = []
    for rank1 in ranks:
        row = []
        for rank2 in ranks:
            tau, p_value = stats.kendalltau(rank1, rank2)
            row.append(tau)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    return df

def set_dif(order_a, order_b):
    s = set(order_a)
    t = set(order_b)
    return len(s^t) / (len(s) + len(t))

def venn(ranks, labels, n_selection):
    matrix = []
    for rank1 in ranks:
        row = []
        for rank2 in ranks:
            v = set_dif(rank1[0:n_selection], rank2[0:n_selection])
            row.append(v)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    return df

def split_targets(df, targ_min, targ_max, target_split, class_label):

    sort_class = []
    min_target = float(targ_min)
    max_target = float(targ_max)

    if not target_split:
        target_split = [max_target]
    if min(target_split) >= min_target and max(target_split) <= max_target:
        complete_split = list(set([min_target] + target_split + [max_target]))
        complete_split.sort()
        for first, second in zip(complete_split, complete_split[1:]):
            sort_class.append('{}_{}'.format(round(first, 3), round(second, 3)))
        sort_class = np.array(sort_class)
    else:
        raise Exception('Out of bounds values for target split: {} must be between {} and {}.'.format(target_split, min_target, max_target))

    target_classes = []
    the_targets = df[class_label].astype(float)
    for sample in the_targets:
        if sample == complete_split[0]:
            target_classes.append(sort_class[0])
        else:
            i = 0    
            for first, second in zip(complete_split, complete_split[1:]):
                if sample > first and sample <= second:
                    target_classes.append(sort_class[i])
                i = i+1
    target_classes = pd.DataFrame(target_classes, index=the_targets.index)
    sort_class = np.sort(target_classes[0].unique())

    return sort_class, target_classes

def shift_target(df, class_label):
    targets_to_return = df[class_label]
    min_threshold = 0.1
    if df[class_label].min() < min_threshold:
        print("\nWARNING: This implementation does not allow negative or zero outputs, converting {} to a range >= {}.\nLabels will retain the original target values.".format(class_label, min_threshold))
        targets_to_return = df[class_label] + (min_threshold - df[class_label].min())
    return targets_to_return