from abc import abstractmethod
import numpy as np
import pandas as pd

from pyod.models.knn import KNN
from pyod.models.iforest import IForest


class OutlierDetection:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        
    @abstractmethod
    def fit_transform(self, df_train, df_corrupted):
        pass


    def cat_out_detect(self, df_train, df_corrupted):
        df_outliers = df_corrupted[self.categorical_columns].copy()
    
        for col in df_train.columns:
            if col in self.categorical_columns:
                vals_train_unique = df_train[col].unique()

                ## add a respective outlier col for each col
                df_outliers[col + "_outlier"] = ''
                
                for i in df_corrupted[col].index:
                    if df_corrupted[col].loc[i] in vals_train_unique:
                        df_outliers[col + "_outlier"].loc[i] = 0
                    else:
                        df_outliers[col + "_outlier"].loc[i] = 1
                
        return df_outliers



class NoOutlierDetection(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()
        
        return df_outliers
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class PyodGeneralOutlierDetection(OutlierDetection):

    def num_out_detect(self, df_train, df_corrupted, pyod_model):
        df_outliers = df_corrupted[self.numerical_columns].copy()
    
        for col in df_train.columns:
            if col in self.numerical_columns:
                ## find indices of records with NaNs in col in df_corrupted
                nan_idx = df_corrupted[df_corrupted[col].isnull()].index
                non_nan_idx = df_corrupted.loc[set(df_corrupted.index) - set(nan_idx)].index
                
                ## pd series -> np column, needs to be 2D array
                ## taking only the non-NaN records in the corrupted data
                col_tr_arr = np.array(df_train[col]).reshape(-1,1)
                col_corr_arr = np.array(df_corrupted.loc[non_nan_idx][col]).reshape(-1,1)

                ## fit the dataset to the model
                model = pyod_model
                model.fit(col_tr_arr)

                ## predict raw anomaly score
                scores_pred = model.decision_function(col_corr_arr) * -1

                ## prediction of a datapoint category outlier or inlier
                y_pred = model.predict(col_corr_arr)

                ## add a respective outlier col for each col
                df_outliers[col + "_outlier"] = ''
                df_outliers[col + "_outlier"].loc[non_nan_idx] = y_pred ## 0: inlier, 1: outlier
                df_outliers[col + "_outlier"].loc[nan_idx] = 0
                
        return df_outliers


    def __call__(self, df_train, df_corrupted):
        return self.num_out_detect(df_train, df_corrupted, pyod_model)


        
class PyODKNN(PyodGeneralOutlierDetection):

    def fit_transform(self, df_train, df_corrupted):
        pyod_model = KNN()
    
        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers[col + "_outlier"].loc[i] == 1:
                    df_outliers[col].loc[i] = np.nan

        df_outliers = df_outliers[df_corrupted.columns]
        
        return df_outliers
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)

    
    
class PyODIsolationForest(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()
        
        feature_transformation = self.feature_transform.fit(df_train)
        x = feature_transformation.transform(df_train).toarray()
        
        model = IForest(contamination=0.25)
        model.fit(x)
        
        xx = feature_transformation.transform(df_outliers).toarray()

        df_outliers["outlier"] = model.predict(xx) ## 0: inlier, 1: outlier
        
        return df_outliers
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



# def pyod_num_out_detect(self, df_train, df_corrupted, pyod_model):
#     df_outliers = df_corrupted[self.numerical_columns].copy()
    
#     for col in df_train.columns:
#         if col in self.numerical_columns:
#             ## find indices of records with NaNs in col in df_corrupted
#             nan_idx = df_corrupted[df_corrupted[col].isnull()].index
#             non_nan_idx = df_corrupted.loc[set(df_corrupted.index) - set(nan_idx)].index
            
#             ## pd series -> np column, needs to be 2D array
#             ## taking only the non-NaN records in the corrupted data
#             col_tr_arr = np.array(df_train[col]).reshape(-1,1)
#             col_corr_arr = np.array(df_corrupted.loc[non_nan_idx][col]).reshape(-1,1)

#             ## fit the dataset to the model
#             model = pyod_model
#             model.fit(col_tr_arr)

#             ## predict raw anomaly score
#             scores_pred = model.decision_function(col_corr_arr) * -1

#             ## prediction of a datapoint category outlier or inlier
#             y_pred = model.predict(col_corr_arr)

#             ## add a respective outlier col for each col
#             df_outliers[col + "_outlier"] = ''
#             df_outliers[col + "_outlier"].loc[non_nan_idx] = y_pred ## 0: inlier, 1: outlier
#             df_outliers[col + "_outlier"].loc[nan_idx] = 0
            
#     return df_outliers 