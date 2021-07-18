import warnings
warnings.filterwarnings('ignore')

import pickle

import random
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from jenga.basis import Dataset
from jenga.corruptions.generic import MissingValues, SwappedValues, CategoricalShift
from jenga.corruptions.numerical import Scaling, GaussianNoise
from jenga.cleaning.ppp import PipelinePerformancePrediction
from jenga.cleaning.outlier_detection import NoOutlierDetection, PyODKNNOutlierDetection, PyODIsolationForestOutlierDetection, PyODPCAOutlierDetection, PyODCBLOFOutlierDetection, PyODSOSOutlierDetection, SklearnOutlierDetection
from jenga.cleaning.imputation import MeanModeImputation, SklearnImputation, DatawigImputation, IterativeImputation
from jenga.cleaning.clean import Clean


## use categorical columns as strings
def cat_cols_to_str(df):
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)

    return df


def run_experiment(dataset_name, learner, param_grid, corruptions, fraction, cleaners, num_repetitions, categorical_precision_threshold=0.85, numerical_std_error_threshold=0.9):
    
    ## dataset
    dataset = Dataset(dataset_name)
    
    all_data = dataset.all_data
    attribute_names = dataset.attribute_names
    attribute_types = dataset.attribute_types
    
    ## categorical and numerical features
    categorical_columns = dataset.categorical_columns
    numerical_columns = dataset.numerical_columns
    print(f"Found {len(categorical_columns)} categorical and {len(numerical_columns)} numeric features \n")
    
    ## train and test data
    df_train, lab_train, df_test, lab_test = dataset.get_train_test_data()
    ### if we don't convert the categorical columns to str, the swapping corruption doesn't let us assign new values to the column: "Cannot setitem on a Categorical with a new category, set the categories first"
    df_train = cat_cols_to_str(df_train)
    df_test = cat_cols_to_str(df_test)
    
    ## pipeline performance prediction (ppp)
    ppp = PipelinePerformancePrediction(df_train, lab_train, df_test, lab_test, categorical_columns, numerical_columns, learner, param_grid)
    ppp_model = ppp.fit_ppp(df_train)
    ## the class column is added to the training data for Autogluon model fit
    if learner == None:
        df_train = df_train.loc[:, df_train.columns != 'class']
    
    ## generate corrupted data
    for _ in range(num_repetitions):
        df_corrupted, perturbations, cols_perturbed, summary_col_corrupt = ppp.get_corrupted(df_test, corruptions, fraction, num_repetitions)
    
    ## cleaning
    clean = Clean(df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold, ppp, ppp_model, cleaners)
    df_outliers, df_cleaned, corrupted_score_ppp, best_cleaning_score, cleaner_scores_ppp, summary_cleaners = clean(df_train, df_test, df_corrupted, cols_perturbed)
    
    ## results
    result = {
        'ppp_score_model': ppp.predict_score_ppp(ppp_model, df_test),
        'ppp_score_corrupted': corrupted_score_ppp,
        'ppp_score_cleaned': best_cleaning_score,
        'ppp_scores_cleaners': cleaner_scores_ppp
    }
    
    ## summary
    summary = {
        'dataset': dataset_name,
        'model': learner,
        'corruptions': summary_col_corrupt,
        'cleaners': summary_cleaners,
        'result': result
    }
    
    return summary #summary_col_corrupt, result


def save_obj(obj, name):
    with open('results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



datasets = ['thoracic_surgery', 'cleve', 'acute-inflammations']


fractions = [0.15, 0.25, 0.5, 0.75, 0.9]


cleaners = []
for od in [NoOutlierDetection, PyODKNNOutlierDetection, PyODIsolationForestOutlierDetection, PyODPCAOutlierDetection, PyODCBLOFOutlierDetection, PyODSOSOutlierDetection, SklearnOutlierDetection]:
    for imp in [MeanModeImputation, SklearnImputation, DatawigImputation, IterativeImputation]:
        cleaners.append((od, imp))


learner = SGDClassifier(loss='log')
param_grid = {
    'learner__max_iter': [500, 1000, 5000],
    'learner__penalty': ['l2', 'l1', 'elasticnet'],
    'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
}


## Gaussian
corruption = [GaussianNoise]
ind_results = {}

for fraction in fractions:
    ind_results[fraction] = []
    for dataset in datasets:
        try:
            ind_results[fraction].append(run_experiment(dataset, learner, param_grid, corruption, fraction, cleaners, 5))
        except ConnectionError:
            print(f'Connection refused for dataset: {dataset}')
            continue
        except ValueError:
            print("Something went wrong with a value :(")
            continue

save_obj(ind_results, "gaussian/ind_results_sgd")