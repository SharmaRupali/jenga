from abc import abstractmethod
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve

from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.sos import SOS



class OutlierDetection:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold):
        
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold

        self.predictors = {}


    def __repr__(self):
        return f"{self.__class__.__name__}"
        
        
    @abstractmethod
    def fit_transform(self, df_train, df_corrupted):
        pass


    ## cast categorical columns as strings to avoid type error (mainly for AutoGluon)
    def cat_cols_to_str(self, df):
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df



class NoOutlierDetection(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()

        ## add a respective outlier col for each col
        for col in df_corrupted.columns:
            df_outliers[col + "_outlier"] = 0
        
        return df_outliers, self.predictors
    
    
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
                df_outliers.loc[non_nan_idx, col + "_outlier"] = y_pred ## 0: inlier, 1: outlier
                df_outliers.loc[nan_idx, col + "_outlier"] = 1
                
        return df_outliers


    ## outlier detection for categorical columns
    def cat_out_detect(self, df_train, df_corrupted):
        df_outliers = df_corrupted[self.categorical_columns].copy()

        ## Swapping between a categoric and numeric variable messes up the categories of the categoric variable, and there
        ## are issues while comparing the values inside the same column
        ## Here, I am finding all the numeric values in the categoric columns, that came after using the SwappedValues corruption
        ## and repacing them with NaNs, they will later be imputed.
        for col in self.categorical_columns:
            idx_to_nan = df_outliers.index[np.where(df_outliers.applymap(np.isreal)[col] == True)]
            if len(idx_to_nan) != 0:
                df_outliers.loc[idx_to_nan, col] = np.nan
    
        for col in df_train.columns:
            if col in self.categorical_columns:
                vals_train_unique = df_train[col].unique()

                ## add a respective outlier col for each col
                df_outliers[col + "_outlier"] = ''
                
                for i in df_corrupted[col].index:
                    if df_corrupted.loc[i, col] in vals_train_unique:
                        df_outliers.loc[i, col + "_outlier"] = 0
                    else:
                        df_outliers.loc[i, col + "_outlier"] = 1
                
        return df_outliers


    def __call__(self, df_train, df_corrupted):
        return self.num_out_detect(df_train, df_corrupted, pyod_model)


        
class PyODKNNOutlierDetection(PyodGeneralOutlierDetection):

    def fit_transform(self, df_train, df_corrupted):
        pyod_model = KNN()
    
        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan
        
        return df_outliers, self.predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)

    
    
class PyODIsolationForestOutlierDetection(PyodGeneralOutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        pyod_model = IForest(contamination=0.25)

        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan
        
        return df_outliers, self.predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class PyODPCAOutlierDetection(PyodGeneralOutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        pyod_model = PCA(contamination=0.25) # n_components = min(n_samples, n_features) default  # n_selected_components = None

        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan
        
        return df_outliers, self.predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class PyODCBLOFOutlierDetection(PyodGeneralOutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        pyod_model = CBLOF(contamination=0.25) # n_clusters = 8 default

        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan
        
        return df_outliers, self.predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class PyODSOSOutlierDetection(PyodGeneralOutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        pyod_model = SOS(contamination=0.25)

        df_outliers_num = self.num_out_detect(df_train, df_corrupted, pyod_model)
        df_outliers_cat = self.cat_out_detect(df_train, df_corrupted)

        df_outliers = df_outliers_num.join(df_outliers_cat, how='inner')

        for col in df_corrupted.columns:
            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan
        
        return df_outliers, self.predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class SklearnOutlierDetection(OutlierDetection):

    def fit_method(self, df_train):
        ## split the train data further into train and test
        df_train, df_test = train_test_split(df_train, test_size=0.2)

        ## preprocessing
        categorical_preprocessing = Pipeline([
            ('mark-missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
        ])

        numeric_preprocessing = Pipeline([
            ('mark_missing', SimpleImputer(strategy='median')),
            ('scaling', StandardScaler())
        ])


        for col in self.categorical_columns + self.numerical_columns:
            if col in self.categorical_columns:
                if len(df_train[col].unique()) > 1:
                    feature_transform = ColumnTransformer(transformers=[
                        ('categorical_features', categorical_preprocessing, list(set(self.categorical_columns) - {col})),
                        ('numeric_features', numeric_preprocessing, self.numerical_columns)
                    ])

                    param_grid = {
                        'learner__n_estimators': [10, 50, 100, 200],
                    }

                    pipeline = Pipeline([
                        ('features', feature_transform),
                        ('learner', GradientBoostingClassifier())
                    ])

                    search = GridSearchCV(pipeline, param_grid, cv=2, verbose=0, n_jobs=-1)
                    self.predictors[col] = search.fit(df_train, df_train[col])

                    print(f'Classifier for col: {col} reached {search.best_score_}')

                    ## precision-recall curves for finding the likelihood thresholds for minimal precision
                    self.predictors[col].thresholds = {}
                    probas = self.predictors[col].predict_proba(df_test)

                    for label_idx, label in enumerate(self.predictors[col].classes_):
                        prec, rec, threshold = precision_recall_curve(df_test[col]==label, probas[:,label_idx], pos_label=True)
                        prec = prec.tolist(); rec = rec.tolist(); threshold = threshold.tolist()
                        threshold_for_min_prec = np.array([elem >= self.categorical_precision_threshold for elem in prec]).nonzero()[0][0] - 1
                        self.predictors[col].thresholds[label] = threshold_for_min_prec

            elif col in self.numerical_columns:
                feature_transform = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, self.categorical_columns),
                    ('numeric_features', numeric_preprocessing, list(set(self.numerical_columns) - {col}))
                ])

                param_grid = {
                    'learner__n_estimators': [10, 50, 100],
                }

                self.predictors[col] = {}

                for perc_name, percentile, in zip(['lower', 'median', 'upper'], [1.0 - self.numerical_std_error_threshold, 0.5, self.numerical_std_error_threshold]):
                    pipeline = Pipeline([
                        ('features', feature_transform),
                        ('learner', GradientBoostingRegressor(loss='quantile', alpha=percentile))
                    ])

                    search = GridSearchCV(pipeline, param_grid, cv=2, verbose=0, n_jobs=-1)
                    self.predictors[col][perc_name] = search.fit(df_train, df_train[col])

                    print(f'Regressor for col: {col}/{perc_name} reached {search.best_score_}')

        return self.predictors


    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()

        ## training
        predictors = self.fit_method(df_train)

        for col in self.categorical_columns + self.numerical_columns:
            if col in self.categorical_columns:
                if col in predictors.keys():
                    y_pred = predictors[col].predict(df_corrupted)
                    y_proba = predictors[col].predict_proba(df_corrupted)

                    for label_idx, label in enumerate(predictors[col].classes_):
                        precision_pred = predictors[col].thresholds[label] <= y_proba[:,label_idx]
                        outliers = precision_pred & (df_corrupted[col] != y_pred)

            elif col in self.numerical_columns:
                lower_percentile = predictors[col]['lower'].predict(df_corrupted)
                upper_percentile = predictors[col]['upper'].predict(df_corrupted)
                outliers = (df_corrupted[col] < lower_percentile) | (df_corrupted[col] > upper_percentile)

            ## find indices of records with NaNs in col in df_corrupted
            nan_idx = df_corrupted[df_corrupted[col].isnull()].index
            non_nan_idx = df_corrupted.loc[set(df_corrupted.index) - set(nan_idx)].index

            ## add a respective outlier col for each col
            df_outliers[col + "_outlier"] = ''
            df_outliers.loc[non_nan_idx, col + "_outlier"] = outliers.astype('int') ## 0: inlier, 1: outlier
            df_outliers.loc[nan_idx, col + "_outlier"] = 1

            for i in df_outliers.index:
                if df_outliers.loc[i, col + "_outlier"] == 1:
                    df_outliers.loc[i, col] = np.nan

            print(f'Column {col} contained {len(nan_idx)} nans before, now {df_outliers[col].isnull().sum()}')

        return df_outliers, predictors
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)