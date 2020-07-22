import pandas as pd

from jenga.cleaning.ppp import PipelinePerformancePrediction
from jenga.cleaning.cleaner import Cleaner
from jenga.cleaning.outlier_detection import NoOutlierDetection, PyODKNN, PyODIsolationForest
from jenga.cleaning.imputation import NoImputation, MeanModeImputation, DatawigImputation



class Clean:
    
    def __init__(self, 
                 df_train, 
                 df_corrupted, 
                 categorical_columns, 
                 numerical_columns,
                 ppp,
                 ppp_model,
                 cleaners):

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
            
        
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


    def get_cleaned(self, df_train, df_corrupted):

        print("\nApplying cleaners... \n")
        
        score_no_cleaning = self.ppp.predict_score_ppp(self.ppp_model, df_corrupted)
        print(f"PPP score no cleaning: {score_no_cleaning}")

        summ_clean = {}
        summary_cleaners = []
        
        cleaner_scores_ppp = []
        for cleaner in self.cleaners:
            df_cleaned = cleaner.apply_cleaner(df_train, df_corrupted, self.categorical_columns, self.numerical_columns)
            cleaner_score = self.ppp.predict_score_ppp(self.ppp_model, df_cleaned)
            # print(f"Outlier detection method: {cleaner.outlier_detection}")
            # print(f"Imputation method: {cleaner.imputation}")
            print(f"PPP score with cleaning: {cleaner}: {cleaner_score} \n")
            cleaner_scores_ppp.append(cleaner_score)

            summ_clean = {"Outlier detection method": cleaner.outlier_detection, "Imputation method": cleaner.imputation, "PPP score with cleaning": cleaner_score}
            summary_cleaners.append(summ_clean) ## saving results for returning individuals too
            
        
        roc_scores_for_best = []
        for i in range(len(cleaner_scores_ppp)):
            roc_scores_for_best.append(cleaner_scores_ppp[i]["roc_auc_acore"])

        best_cleaning_idx = pd.Series(roc_scores_for_best).idxmax()
        best_cleaning_score = cleaner_scores_ppp[best_cleaning_idx]

        df_cleaned = self.cleaners[best_cleaning_idx].apply_cleaner(df_train, df_corrupted, self.categorical_columns, self.numerical_columns)
        print(f"Best cleaning method:")
        print(f"Cleaning score: {self.cleaners[best_cleaning_idx]}: {best_cleaning_score} \n\n\n\n")

        if best_cleaning_score["roc_auc_acore"] > score_no_cleaning["roc_auc_acore"]:
            print("Cleaning improved the score \n\n\n\n")
            # df_cleaned = self.cleaners[best_cleaning_idx].apply_cleaner(df_train, df_corrupted, self.categorical_columns, self.numerical_columns)
            # print(f"Best cleaning method:")
            # # print(f"Outlier detection method: {self.cleaners[best_cleaning_idx].outlier_detection}")
            # # print(f"Imputation method: {self.cleaners[best_cleaning_idx].imputation}")
            # print(f"Cleaning score: {self.cleaners[best_cleaning_idx]}: {best_cleaning_score} \n\n\n\n")
        else:
            print("Cleaning didnt't improve the score \n\n\n\n")
            
        return df_cleaned, score_no_cleaning, best_cleaning_score, cleaner_scores_ppp, summary_cleaners
    
    
    def __call__(self, df_train, df_corrupted):
        return self.get_cleaned(df_train, df_corrupted)