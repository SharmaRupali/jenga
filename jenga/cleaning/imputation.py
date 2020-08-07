from abc import abstractmethod
import numpy as np
import pandas as pd

# import datawig
from autogluon import TabularPrediction as task



class Imputation:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        
    def __repr__(self):
        return f"{self.__class__.__name__}"


    @abstractmethod
    def fit_transform(self, df_train, df_corrupted, predictors):
        pass


    def cat_cols_to_str(self, df):
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    
    
class NoImputation(Imputation):    
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):        
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit_transform(self, df_train, df_corrupted, predictors):
        df_imputed = df_corrupted.copy()
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)
    
    
    
class MeanModeImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.means = {}
        self.modes = {}
    
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit_transform(self, df_train, df_corrupted, predictors):
        df_imputed = df_corrupted.copy()
        
        for col in df_train.columns:
            if col in self.numerical_columns:
                # mean imputer
                mean = np.mean(df_train[col])
                self.means[col] = mean
            elif col in self.categorical_columns:
                # mode imputer
                mode = df_train[col].value_counts().index[0]
                self.modes[col] = mode
                
                
        for col in df_corrupted.columns:
            if col in self.numerical_columns:
                # mean imputer
                df_imputed[col].fillna(self.means[col], inplace=True)
            elif col in self.categorical_columns:
                # mode imputer
                df_imputed[col].fillna(self.modes[col], inplace=True)
                
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)



class AutoGluonImputation(Imputation):

    def fit_transform(self, df_train, df_corrupted, predictors):
        df_train = self.cat_cols_to_str(df_train)
        df_imputed = self.cat_cols_to_str(df_corrupted.copy())

        if not predictors:
            for col in self.categorical_columns:
                predictors[col] = task.fit(train_data=df_train, label=col, problem_type='multiclass')

            for col in self.numerical_columns:
                predictors[col] = task.fit(train_data=df_train, label=col, problem_type='regression')


        for col in df_corrupted.columns:
            df_imputed[col + '_imputed'] = predictors[col].predict(df_imputed.drop([col], axis=1)) # drop the actual column before predicting
            perf = predictors[col].evaluate_predictions(df_imputed[col], df_imputed[col + '_imputed'], auxiliary_metrics=False)

            df_imputed[col].fillna(df_imputed[col + '_imputed'], inplace=True)


        df_imputed = df_imputed[df_corrupted.columns]

        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)