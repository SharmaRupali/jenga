import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from jenga.models.model import Model
from jenga.corruptions.perturbations import Perturbation


class PipelinePerformancePrediction:
    
    def __init__(self, seed, dataset, categorical_columns, numerical_columns, learner, param_grid, pipeline=None):
        
        self.learner = learner
        self.param_grid = param_grid
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        
        ## get training and test sets
        self.train_data, self.train_labels, self.test_data, self.test_labels = dataset.get_train_test_data()
        
        
        ## define preprocessing pipeline if not given
        if pipeline == None:
            # preprocessing pipeline for numerical columns
            transformer_numeric = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('standard_scale', StandardScaler())
            ])

            # preprocessing pipeline for categorical columns
            transformer_categorical = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='__NA__')),
                ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
            ])

            # preprocessor
            feature_transform = ColumnTransformer(transformers=[
                ('categorical_features', transformer_categorical, self.categorical_columns),
                ('numerical_features', transformer_numeric, self.numerical_columns)
            ])

            ## prediction pipeline: append classifier (learner) to the preprocessing pipeline
            self.pipeline = Pipeline([
                ('features', feature_transform),
                ('learner', self.learner)
            ])
        else:
            self.pipeline = pipeline
            
        
        ## get model components
        self.model_obj = Model(seed, 
                          self.train_data, 
                          self.train_labels, 
                          self.test_data, 
                          self.test_labels, 
                          self.categorical_columns, 
                          self.numerical_columns, 
                          self.pipeline, 
                          self.learner, 
                          self.param_grid)
        
        
    def get_corrupted(self, df):
        
        print(f"Generating corrupted training data on {len(df)} rows...")
        
        # corruption perturbations to apply
        corr_perturbations = Perturbation(self.categorical_columns, self.numerical_columns)
        df_corrupted, perturbations, cols_perturbed = corr_perturbations.apply_perturbation(df, 5)
        
        return df_corrupted, perturbations, cols_perturbed
        
        
    def fit_ppp(self, df):
        
        model = self.model_obj.fit_model(df, self.train_labels)
        
        return model
    
    
    def predict_score_ppp(self, model, df):
        
        score = self.model_obj.roc_auc_score_on_test_data(model, df)
        
        return score