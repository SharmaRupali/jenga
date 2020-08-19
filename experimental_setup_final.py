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
from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import Scaling, GaussianNoise
from jenga.cleaning.ppp import PipelinePerformancePrediction
from jenga.cleaning.outlier_detection import NoOutlierDetection, PyODKNNOutlierDetection, PyODIsolationForestOutlierDetection, AutoGluonOutlierDetection
from jenga.cleaning.imputation import NoImputation, MeanModeImputation, AutoGluonImputation
from jenga.cleaning.clean import Clean


seed = 10


def run_experiment(dataset_name, learner, param_grid, corruptions, fraction, cleaners, num_repetitions, categorical_precision_threshold=0.7, numerical_std_error_threshold=2.0):
    
    ## dataset
    dataset = Dataset(seed, dataset_name)
    
    all_data = dataset.all_data
    attribute_names = dataset.attribute_names
    attribute_types = dataset.attribute_types
    
    ## categorical and numerical features
    categorical_columns = dataset.categorical_columns
    numerical_columns = dataset.numerical_columns
    print(f"Found {len(categorical_columns)} categorical and {len(numerical_columns)} numeric features \n")
    
    ## train and test data
    df_train, lab_train, df_test, lab_test = dataset.get_train_test_data()
    
    
    ## pipeline performance prediction (ppp)
    ppp = PipelinePerformancePrediction(seed, df_train, lab_train, df_test, lab_test, categorical_columns, numerical_columns, learner, param_grid)
    ppp_model = ppp.fit_ppp(df_train)
    
    ## generate corrpted data
    df_corrupted, perturbations, cols_perturbed, summary_col_corrupt = ppp.get_corrupted(df_test, corruptions, fraction, num_repetitions)
    
    ## cleaning
    clean = Clean(df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold, ppp, ppp_model, cleaners)
    df_cleaned, corrupted_score_ppp, best_cleaning_score, cleaner_scores_ppp, summary_cleaners = clean(df_train, df_corrupted)
    
    ## results
    result = {
        'ppp_score_model': ppp.predict_score_ppp(ppp_model, df_test),
        'ppp_score_corrupted': corrupted_score_ppp,
        'ppp_score_cleaned': best_cleaning_score,
        'ppp_scores_cleaners': cleaner_scores_ppp
    }
#     print('\n'.join([f'{key}:{val}' for key, val in result.items()]))
    
    ## summary
    summary = {
        'dataset': dataset_name,
        'model': learner,
        'corruptions': summary_col_corrupt,
        'cleaners': summary_cleaners,
        'result': result
    }
#     print('\n\n\n\n'.join([f'{key}:{val}' for key, val in summary.items()]))
    
    return summary #summary_col_corrupt, result


datasets = ['parkinsons', 'heart-statlog', 'credit-g']


## model parameters
## models is a dict where key = leaner & value = param_grid
models = {SGDClassifier(loss='log'): {'learner__max_iter': [500, 1000, 5000], 
                                         'learner__penalty': ['l2', 'l1', 'elasticnet'], 
                                         'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
                                        }, 
          RandomForestClassifier():{'learner__n_estimators': [100, 200, 500], 
                                    'learner__max_depth': [5, 10, 15]
                                   }
         }

## make dict of multiple leraners and corresponding param_grids


corruptions = [MissingValues, SwappedValues, Scaling, GaussianNoise]

fractions = [0.15, 0.25, 0.5, 0.75, 0.9]

cleaners = [
    (NoOutlierDetection, MeanModeImputation),
    (PyODKNNOutlierDetection, MeanModeImputation),
    (PyODKNNOutlierDetection, AutoGluonImputation),
    (PyODIsolationForestOutlierDetection, MeanModeImputation),
    (PyODIsolationForestOutlierDetection, AutoGluonImputation),
    (AutoGluonOutlierDetection, MeanModeImputation),
    (AutoGluonOutlierDetection, AutoGluonImputation)
]


for _ in range(10):
  print("\n\n..................................ITERATION..................................\n")
  ind_results = []

  for dataset in datasets:
      for learner, param_grid in models.items():
          for fraction in fractions:
              ind_results.append(run_experiment(dataset, learner, param_grid, [MissingValues], fraction, cleaners, 100))


ind_results
