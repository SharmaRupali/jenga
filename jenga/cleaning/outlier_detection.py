from abc import abstractmethod

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pyod.models.knn import KNN
from pyod.models.iforest import IForest


class OutlierDetection:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        
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
        self.feature_transform = ColumnTransformer(transformers=[
            ('categorical_features', transformer_categorical, self.categorical_columns),
            ('numerical_features', transformer_numeric, self.numerical_columns)
        ], sparse_threshold=1.0)
        
        
        @abstractmethod
        def fit_transform(self, df_train, df_corrupted):
            pass



class NoOutlierDetection(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()
        
        return df_outliers


        
class PyODKNN(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()
        
        feature_transformation = self.feature_transform.fit(df_train)
        x = feature_transformation.transform(df_train).toarray()
        
        model = KNN()
        model.fit(x)
        
        xx = feature_transformation.transform(df_outliers).toarray()

        df_outliers["outlier"] = model.predict(xx) ## 0: inlier, 1: outlier
        
        return df_outliers

    
    
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
