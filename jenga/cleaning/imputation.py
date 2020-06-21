from abc import abstractmethod
import numpy as np
import pandas as pd

import datawig



class Imputation:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
    
    @abstractmethod
    def fit(self, df_train):
        pass
    
    @abstractmethod
    def transform(self, df_corrupted):
        pass
    
    @abstractmethod
    def fit_transform(self, df_train, df_corrupted):
        pass
		
	
	
class MeanModeImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.means = {}
        self.modes = {}
    
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit(self, df_train):
        for col in df_train.columns:
            if col in self.numerical_columns:
                # mean imputer
                mean = np.mean(df_train[col])
                self.means[col] = mean
            elif col in self.categorical_columns:
                # mode imputer
                mode = df_train[col].value_counts().index[0]
                self.modes[col] = mode
                
                
    def transform(self, df_corrupted):
        df_imputed = df_corrupted.copy()
        
        for col in df_corrupted.columns:
            if col in self.numerical_columns:
                # mean imputer
                df_imputed[col].fillna(self.means[col], inplace=True)
            elif col in self.categorical_columns:
                # mode imputer
                df_imputed[col].fillna(self.modes[col], inplace=True)
                
        return df_imputed
		
		
		
class DatawigImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        self.model = None
        
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns)
    
    
    def fit_transform(self, df_train, df_corrupted):
        df_imputed = df_corrupted.copy()

        for col in df_train.columns:
            if pd.api.types.is_categorical_dtype(df_train[col]):
                df_train[col] = df_train[col].astype(str)

        for col in df_corrupted.columns:
            if pd.api.types.is_categorical_dtype(df_corrupted[col]):
                df_corrupted[col] = df_corrupted[col].astype(str)


        for col in self.categorical_columns + self.numerical_columns:
            output_column = col
            input_columns = list(set(df_train.columns) - set([output_column]))

            print(f"Fitting model for column: {col}")
            model = datawig.SimpleImputer(input_columns, output_column, 'imputer_model')
            model.fit(df_train)

            df_imputed = model.predict(df_imputed)
            df_imputed[col].fillna(df_imputed[col + '_imputed'], inplace=True)
            df_imputed = df_imputed[df_corrupted.columns]

        return df_imputed