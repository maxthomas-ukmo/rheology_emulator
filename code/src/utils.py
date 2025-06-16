# TODO: Make definition of features include t and t+1 variables for everything except the target
# so like siconc(t), siconc(t+1), sivelv(t) -> sivelv(t+1)
# 
import argparse as ap
import pickle
import random
import warnings
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# Parse command line arguments
def parse_args(dummy_args=False):
    '''Parse command line arguments.
    
    Args: 
        dummy_args: bool, if True, return dummy arguments for testing
        
    Returns:
        args: dict, dictionary of arguments
    '''
    if dummy_args:
        args = {'model_type': 'SGDRegressor',
            'test_fraction': 0.1,
            'suite': 'u-cn464',
            'version': 'raw_v0',
            'features': ['siconc', 'sithic', 'utau_ai', 'utau_oi', 'vtau_ai', 'vtau_oi'],#,'sishea', 'sistre'],
            'labels': ['sig1_pnorm'],
            'flatten': True,
            'StandardScalar': True
               }
        return args
    else:
        parser = ap.ArgumentParser(description='Train a regression model')
        parser.add_argument('--model_type', type=str, default='SGDRegressor', help='Type of model to use')
        parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of data to use for testing')
        parser.add_argument('--suite', type=str, default='u-cn464', help='Name of suite to use')
        parser.add_argument('--version', type=str, default='raw_v0', help='Version of suite to use')
        parser.add_argument('--features', type=str, default='siconc,sithic,utau_ai,utau_oi,vtau_ai,vtau_oi,sishea,sistre,sig1_pnorm,sig2_pnorm,sivelv,sivelu', help='List of features to use')
        parser.add_argument('--labels', type=str, default='sig1_pnorm', help='List of labels to use')
        parser.add_argument('--flatten', type=bool, default=True, help='Flatten data')
        parser.add_argument('--StandardScalar', type=bool, default=True, help='Standardise data')
        parser.add_argument('--tune', type=bool, default=True, help='Tune hyperparameters')
        parser.add_argument('--data_points', type=int, default=1000, help='Data points to feed model (less for quick testing)')
        parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
        parser.add_argument('--validation_fraction', type=float, default=0.2, help='Fraction of train data to validate on')

        args = parser.parse_args()
        args.features = args.features.split(',')     
        args.labels = args.labels.split(',')         
                                                                                                                                                                                                                                                     
        return vars(args)

# Load data function
def load_data(args):
    '''
    Load raw data and make feature (X(t)) - label (y(t+1)) pairs.
    '''
    suite = args['suite']
    version = args['version']
    filename = '../data/' + suite + '/raw/' + suite + '_' + version + '.nc'
    pairs = make_feature_label_pairs(filename, args['features'], args['labels'], flatten=args['flatten'])
    return pairs

def make_feature_label_pairs(filename, feature_names, label_names, flatten=False, feature_names_tp1=None):
    data = xr.open_dataset(filename)
    if flatten:
        data = data.stack(xy=('x', 'y'))
        data = data.dropna(dim='xy')
    features = data[feature_names]
    # if feature_names_tp1 is not None:
    #     features_tp1 = data[feature_names_tp1]
    labels = data[label_names]
    pairs = []
    for itime in range(features.time_counter.size - 1):
        feature = features.isel(time_counter=itime)
        # if feature_names_tp1 is not None:
        #     feature_tp1 = features_tp1.isel(time_counter=itime+1)
        #     # add feature_tp1 to feature as extra variables
        #     feature = xr.concat([feature, feature_tp1], dim='time_counter')
        label = labels.isel(time_counter=itime+1)
        pairs.append((feature, label))
    return pairs

# Make test and train sets
def make_test_train(pairs, args):
    '''
    Split pairs into training and testing sets, and concatonate the pairs into one dataframe.

    Args:
        pairs: list, list of pairs of feature and label dataframes
        args: dict, dictionary of arguments

    Returns:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        test_X: dataframe, testing feature data
        test_y: dataframe, testing label data
    '''
    pairs_df = []
    for pair in pairs:
        try:
            pairs_df.append( 
                            (pd.DataFrame(pair[0].to_array().values.T), 
                            pd.DataFrame(pair[1].to_array().values.reshape(-1, ))) 
                            )
        except:
            pairs_df.append( 
                            (pd.DataFrame(pair[0].to_array().values.T), 
                            pd.DataFrame(pair[1].values.reshape(-1, )))
            )

    # randomly select 10% of the pairs as a test holdout
    n = len(pairs_df)
    n_test = int(args['test_fraction']*n)
    test_indices = random.sample(range(n), n_test)
    train_indices = [i for i in range(n) if i not in test_indices]

    train_pairs = [pairs_df[i] for i in train_indices]
    test_pairs = [pairs_df[i] for i in test_indices]

    train_X = pd.concat([pair[0] for pair in train_pairs], axis=0)
    train_y = pd.concat([pair[1] for pair in train_pairs], axis=0)
    test_X = pd.concat([pair[0] for pair in test_pairs], axis=0)
    test_y = pd.concat([pair[1] for pair in test_pairs], axis=0)
    
    print('Train shape initial: ', train_X.shape, train_y.shape)
    print('Test shape initial: ', test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y


def reduce_data_points(train_X, train_y, n_data_points):
    '''
    Reduce the number of data points in the training set for ease of model development. TODO: do this better with batch training

    Args:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        n_data_points: int, number of data points to keep

    Returns:
        train_X: dataframe, training feature data reduced
        train_y: dataframe, training label data reduced
    '''
    if n_data_points > 0:
        indicies = random.sample(range(len(train_X)), n_data_points)
        train_X = train_X.iloc[indicies]
        train_y = train_y.iloc[indicies]
    return train_X, train_y

def make_validation(train_X, train_y, validation_fraction):
    '''
    Make validation set from training set.

    Args:
        train_X: dataframe, training feature data
        train_y: dataframe, training label data
        validation_fraction: float, fraction of data to use for validation

    Returns:
        train_X: dataframe, training feature data less validation data
        train_y: dataframe, training label data less validation data
        val_X: dataframe, validation feature data
        val_y: dataframe, validation label data
    '''
    n = len(train_X)
    n_val = int(validation_fraction*n)
    val_indices = random.sample(range(n), n_val)
    train_indices = [i for i in range(n) if i not in val_indices]

    val_X = train_X.iloc[val_indices]
    val_y = train_y.iloc[val_indices]
    train_X = train_X.iloc[train_indices]
    train_y = train_y.iloc[train_indices]

    print('Train shape reduced: ', train_X.shape, train_y.shape)
    print('Validation shape: ', val_X.shape, val_y.shape)

    return train_X, train_y, val_X, val_y