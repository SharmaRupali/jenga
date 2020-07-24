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
    def fit_transform(self, df_train, df_corrupted):
        pass


    def cat_cols_to_str(self, df):
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    
    
class NoImputation(Imputation):    
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):        
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit_transform(self, df_train, df_corrupted):
        df_imputed = df_corrupted.copy()
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)
    
    
    
class MeanModeImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.means = {}
        self.modes = {}
    
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit_transform(self, df_train, df_corrupted):
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
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)

    

# class DatawigImputation(Imputation):
    
#     def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):        
#         Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
#     def fit_transform(self, df_train, df_corrupted):
#         df_imputed = df_corrupted.copy()

#         for col in df_train.columns:
#             if pd.api.types.is_categorical_dtype(df_train[col]):
#                 df_train[col] = df_train[col].astype(str)

#         for col in df_imputed.columns:
#             if pd.api.types.is_categorical_dtype(df_imputed[col]):
#                 df_imputed[col] = df_imputed[col].astype(str)


#         for col in self.categorical_columns + self.numerical_columns:
#             output_column = col
#             input_columns = list(set(df_train.columns) - set([output_column]))

#             print(f"Fitting model for column: {col}")
#             model = datawig.SimpleImputer(input_columns, output_column, 'imputer_model')
#             model.fit(df_train)

#             df_imputed = model.predict(df_imputed)
#             df_imputed[col].fillna(df_imputed[col + '_imputed'], inplace=True)
#             df_imputed = df_imputed[df_corrupted.columns]

#         return df_imputed
    
    
#     def __call__(self, df_train, df_corrupted):
#         return self.fit_transform(df_train, df_corrupted)



class AutoGluonImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)

        self.df_train = self.cat_cols_to_str(self.df_train)
        self.df_corrupted = self.cat_cols_to_str(self.df_corrupted)

        self.predictors = {}
    
    
    def fit_transform(self, df_train, df_corrupted):
        df_imputed = self.df_corrupted.copy()

        for col in self.categorical_columns:
            self.predictors[col] = task.fit(train_data=self.df_train, label=col, problem_type='multiclass')
            
        for col in self.numerical_columns:
            self.predictors[col] = task.fit(train_data=self.df_train, label=col, problem_type='regression')


        for col in self.df_corrupted.columns:
            df_imputed[col + '_imputed'] = self.predictors[col].predict(df_imputed.drop([col], axis=1)) # drop the actual column before predicting
            perf = self.predictors[col].evaluate_predictions(df_imputed[col], df_imputed[col + '_imputed'], auxiliary_metrics=True)

            df_imputed[col].fillna(df_imputed[col + '_imputed'], inplace=True)
        
        df_imputed = df_imputed[self.df_corrupted.columns]
                
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)