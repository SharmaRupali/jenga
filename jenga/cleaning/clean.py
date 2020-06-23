import pandas as pd

from jenga.cleaning.ppp import PipelinePerformancePrediction
from jenga.cleaning.cleaner import Cleaner
from jenga.cleaning.outlier_detection import NoOutlierDetection, PyODKNN, PyODIsolationForest
from jenga.cleaning.imputation import NoImputation, MeanModeImputation, DatawigImputation


DEFAULT_CLEANERS = [
    (NoOutlierDetection, NoImputation),
    (NoOutlierDetection, MeanModeImputation),
    (NoOutlierDetection, DatawigImputation),
    (PyODKNN, NoImputation),
    (PyODKNN, MeanModeImputation),
    (PyODKNN, DatawigImputation),
    (PyODIsolationForest, NoImputation),
    (PyODIsolationForest, MeanModeImputation),
    (PyODIsolationForest, DatawigImputation)
]


class Clean:
    
    def __init__(self, 
                 df_train, 
                 df_corrupted, 
                 categorical_columns, 
                 numerical_columns,
                 ppp,
                 ppp_model,
                 cleaners=DEFAULT_CLEANERS):

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        self.ppp = ppp
        self.ppp_model = ppp_model
        
        self.cleaners = []
        for outd, imp in cleaners:
            self.cleaners.append(Cleaner(df_train,
                                         df_corrupted,
                                         self.categorical_columns,
                                         self.numerical_columns,
                                         outlier_detection = outd(df_train,
                                                                  df_corrupted,
                                                                  self.categorical_columns,
                                                                  self.numerical_columns),
                                         imputation = imp(df_train,
                                                          df_corrupted,
                                                          self.categorical_columns,
                                                          self.numerical_columns)
                                        )
                                )
            
        
    def get_cleaned(self, df_train, df_corrupted):
        
        score_no_cleaning = self.ppp.predict_score_ppp(self.ppp_model, df_corrupted)
        print(f"PPP score no cleaning: {score_no_cleaning}")
        
        cleaner_scores_ppp = []
        for cleaner in self.cleaners:
            df_cleaned = cleaner.apply_cleaner(df_train, df_corrupted, self.categorical_columns, self.numerical_columns)
            cleaner_score = self.ppp.predict_score_ppp(self.ppp_model, df_cleaned)
            print(f"PPP score with cleaning: {cleaner}: {cleaner_score}")
            cleaner_scores_ppp.append(cleaner_score)
            
        best_cleaning_idx = pd.Series(cleaner_scores_ppp).idxmax()
        best_cleaning_score = cleaner_scores_ppp[best_cleaning_idx]
        if best_cleaning_score > score_no_cleaning:
            df_cleaned = self.cleaners[best_cleaning_idx].apply_cleaner(df_train, df_corrupted, self.categorical_columns, self.numerical_columns)
            print(f"Best cleaning method: {self.cleaners[best_cleaning_idx]}: {best_cleaning_score}")
        else:
            print("Cleaning didnt't improve the score")
            
        return df_cleaned, score_no_cleaning, cleaner_scores_ppp
    
    
    def __call__(self, df_train, df_corrupted):
        return self.get_cleaned(df_train, df_corrupted)