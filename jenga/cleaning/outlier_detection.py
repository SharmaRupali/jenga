from abc import abstractmethod
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pyod.models.knn import KNN
from pyod.models.iforest import IForest

from autogluon import TabularPrediction as task


class OutlierDetection:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold):
        
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold

        self.predictors = {}
        self.predictable_cols = {}


    def __repr__(self):
        return f"{self.__class__.__name__}"
        
        
    @abstractmethod
    def fit_transform(self, df_train, df_corrupted):
        pass


    ## outlier detection for categorical columns (mainly used for PyOD)
    def cat_out_detect(self, df_train, df_corrupted):
        df_outliers = df_corrupted[self.categorical_columns].copy()
    
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


    ## cast categorical columns as strings to avoid type error (mainly for AutoGluon)
    def cat_cols_to_str(self, df):
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df



class NoOutlierDetection(OutlierDetection):
    
    def fit_transform(self, df_train, df_corrupted):
        df_outliers = df_corrupted.copy()
        
        return df_outliers, self.predictors, self.predictable_cols
    
    
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
                df_outliers.loc[nan_idx, col + "_outlier"] = 0
                
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

        df_outliers = df_outliers[df_corrupted.columns]
        
        return df_outliers, self.predictors, self.predictable_cols
    
    
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

        df_outliers = df_outliers[df_corrupted.columns]
        
        return df_outliers, self.predictors, self.predictable_cols
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)



class AutoGluonOutlierDetection(OutlierDetection):

    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold):
        OutlierDetection.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold)

        self.df_train = self.cat_cols_to_str(self.df_train)


        df_train, df_test = train_test_split(self.df_train, test_size=0.2)

        for col in self.categorical_columns:
            self.predictors[col] = task.fit(train_data=df_train, label=col, problem_type='multiclass')

            y_test = df_test[col].dropna() # take only the non-nan records # test_data? OR split the train_data again into train and test
            y_pred = self.predictors[col].predict(df_test.drop([col], axis=1)) # drop the actual column before predicting

            perf = self.predictors[col].evaluate_predictions(y_test, y_pred, auxiliary_metrics=True)

            labels = [k for k in perf['classification_report'].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

            high_precision_labels = []
            for label in labels:
                if perf['classification_report'][label]['precision'] > self.categorical_precision_threshold:
                    high_precision_labels.append(label)

            if high_precision_labels:
                self.predictable_cols[col] = high_precision_labels


        for col in self.numerical_columns:
            self.predictors[col] = task.fit(train_data=df_train, label=col, problem_type='regression')

            y_test = df_test[col].dropna() # take only the non-nan records # test_data? OR split the train_data again into train and test
            y_pred = self.predictors[col].predict(df_test.drop([col], axis=1)) # drop the actual column before predicting

            perf = self.predictors[col].evaluate_predictions(y_test, y_pred, auxiliary_metrics=True)

            if perf['root_mean_squared_error'] < self.numerical_std_error_threshold * y_test.std():
                self.predictable_cols[col] = perf['root_mean_squared_error']


        print(f"Categorical precision threshold: {categorical_precision_threshold}")
        print(f"Numerical Std Error threshold: {numerical_std_error_threshold}")
        print(f"Predictors: {self.predictors}")
        print(f"Predictable Columns: {self.predictable_cols}")


    def fit_transform(self, df_train, df_corrupted):
        df_outliers = self.cat_cols_to_str(df_corrupted.copy())

        presumably_wrong = {}

        for col in self.predictable_cols:
            y_pred = self.predictors[col].predict(df_outliers)
            y_test = df_outliers[col]

            auxiliary_df_test_pred = pd.DataFrame(y_test)
            auxiliary_df_test_pred["pred"] = y_pred

            num_nans = df_outliers[col].isnull().sum()

            if col in self.categorical_columns:
                presumably_wrong_aux = []
                for i in auxiliary_df_test_pred.index:
                    if any(np.isin(self.predictable_cols[col], auxiliary_df_test_pred.loc[i, "pred"])) & (auxiliary_df_test_pred.loc[i, col] != auxiliary_df_test_pred.loc[i, "pred"]):
                        presumably_wrong_aux.append(i)

                presumably_wrong[col] = np.array(presumably_wrong_aux)


            if col in self.numerical_columns:
                presumably_wrong_aux = []
                predictor_rmse = self.predictable_cols[col]
                for i in auxiliary_df_test_pred.index:
                    rmse = np.sqrt((auxiliary_df_test_pred.loc[i, "pred"] - auxiliary_df_test_pred.loc[i, col]) ** 2)
                    if rmse > predictor_rmse * self.numerical_std_error_threshold:
                        presumably_wrong_aux.append(i)

                presumably_wrong[col] = np.array(presumably_wrong_aux)


            for i in presumably_wrong[col]:
                df_outliers.loc[i, col] = np.nan

            print(f"Column {col}: Num NaNs: Before: {num_nans}, Now: {df_outliers[col].isnull().sum()}")

        return df_outliers, self.predictors, self.predictable_cols
    
    
    def __call__(self, df_train, df_corrupted):
        return self.fit_transform(df_train, df_corrupted)