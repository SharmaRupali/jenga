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

from jenga.cleaning.outlier_detection import SklearnOutlierDetection


class Imputation:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold):
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        
        
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
    
    def fit_transform(self, df_train, df_corrupted, predictors):
        df_imputed = df_corrupted.copy()
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)
    
    
    
class MeanModeImputation(Imputation):  
    
    def fit_transform(self, df_train, df_corrupted, predictors):
        df_imputed = df_corrupted.copy()

        means = {}
        modes = {}
        
        for col in df_train.columns:
            if col in self.numerical_columns:
                # mean imputer
                mean = np.mean(df_train[col])
                means[col] = mean
            elif col in self.categorical_columns:
                # mode imputer
                mode = df_train[col].value_counts().index[0]
                modes[col] = mode
                
                
        for col in df_corrupted.columns:
            if col in self.numerical_columns:
                # mean imputer
                df_imputed[col].fillna(means[col], inplace=True)
            elif col in self.categorical_columns:
                # mode imputer
                df_imputed[col].fillna(modes[col], inplace=True)
                
        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)



class SklearnImputation(Imputation):

    def fit_transform(self, df_train, df_corrupted, predictors):
        df_imputed = df_corrupted.copy()
        
        if not predictors:
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
                    predictors[col] = search.fit(df_train, df_train[col])

                    print(f'Classifier for col: {col} reached {search.best_score_}')

                    ## precision-recall curves for finding the likelihood thresholds for minimal precision
                    predictors[col].thresholds = {}
                    probas = predictors[col].predict_proba(df_test)

                    for label_idx, label in enumerate(predictors[col].classes_):
                        prec, rec, threshold = precision_recall_curve(df_test[col]==label, probas[:,label_idx], pos_label=True)
                        prec = prec.tolist(); rec = rec.tolist(); threshold = threshold.tolist()
                        threshold_for_min_prec = np.array([elem >= self.categorical_precision_threshold for elem in prec]).nonzero()[0][0] - 1
                        predictors[col].thresholds[label] = threshold_for_min_prec

                elif col in self.numerical_columns:
                    feature_transform = ColumnTransformer(transformers=[
                        ('categorical_features', categorical_preprocessing, self.categorical_columns),
                        ('numeric_features', numeric_preprocessing, list(set(self.numerical_columns) - {col}))
                    ])

                    param_grid = {
                        'learner__n_estimators': [10, 50, 100],
                    }

                    predictors[col] = {}

                    for perc_name, percentile, in zip(['lower', 'median', 'upper'], [1.0 - self.numerical_std_error_threshold, 0.5, self.numerical_std_error_threshold]):
                        pipeline = Pipeline([
                            ('features', feature_transform),
                            ('learner', GradientBoostingRegressor(loss='quantile', alpha=percentile))
                        ])

                        search = GridSearchCV(pipeline, param_grid, cv=2, verbose=0, n_jobs=-1)
                        predictors[col][perc_name] = search.fit(df_train, df_train[col])

                        print(f'Regressor for col: {col}/{perc_name} reached {search.best_score_}')


        for col in self.categorical_columns + self.numerical_columns:
            prior_missing = df_imputed[col].isnull().sum()

            if prior_missing > 0:
                if col in self.categorical_columns:
                    df_imputed.loc[df_imputed[col].isnull(), col] = predictors[col].predict(df_imputed[df_imputed[col].isnull()])
                elif col in self.numerical_columns:
                    df_imputed.loc[df_imputed[col].isnull(), col] = predictors[col]['median'].predict(df_imputed[df_imputed[col].isnull()])

                print(f'Imputed {prior_missing} values in column {col}')

        return df_imputed
    
    
    def __call__(self, df_train, df_corrupted, predictors):
        return self.fit_transform(df_train, df_corrupted, predictors)