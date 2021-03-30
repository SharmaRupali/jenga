import random
import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


class Model:
    
    def __init__(self,
                 df_train, 
                 lab_train, 
                 df_test, 
                 lab_test, 
                 categorical_columns, 
                 numerical_columns,
                 pipeline,
                 learner, 
                 param_grid):
        
        ## train and test data and labels
        self.df_train = df_train
        self.lab_train = lab_train
        self.df_test = df_test
        self.lab_test = lab_test
        
        ## information about the column types in the dataset
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        ## preprocessing pipeline
        self.pipeline = pipeline
        
        ## information for model parameters
        self.learner = learner
        self.param_grid = param_grid
        
    
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
        

    # method for training a model on the raw data with preprocessing
    def fit_model(self, df_train, lab_train):
        if self.learner != None:
            grid_search = GridSearchCV(self.pipeline, self.param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
            model = grid_search.fit(df_train, lab_train)
        else:
            df_train["class"] = lab_train
            model = TabularPredictor(label="class").fit(df_train)

        return model


    # method for computing evaluation metrics
    def evaluation_metrics(self, model, df_test):
        y_pred = model.predict(df_test)

        if self.learner != None:
            eval_scores = {
                'roc_auc_score': roc_auc_score(self.lab_test, np.transpose(model.predict_proba(df_test))[1]),
                'classification_report': classification_report(self.lab_test, y_pred, output_dict=True)
            }
        else:
            perf = model.evaluate_predictions(y_true=pd.Series(self.lab_test), y_pred=y_pred, auxiliary_metrics=True)
            eval_scores = {
                'roc_auc_score': roc_auc_score(self.lab_test, np.transpose(model.predict_proba(df_test)).to_numpy()[1]),
                'classification_report': perf["classification_report"]
            }

        return eval_scores