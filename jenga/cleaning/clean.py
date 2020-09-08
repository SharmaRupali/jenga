import pandas as pd

from jenga.cleaning.ppp import PipelinePerformancePrediction
from jenga.cleaning.cleaner import Cleaner



class Clean:
    
    def __init__(self, 
                 df_train, 
                 df_corrupted, 
                 categorical_columns, 
                 numerical_columns, 
                 categorical_precision_threshold, 
                 numerical_std_error_threshold,
                 ppp,
                 ppp_model,
                 cleaners):

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        
        self.ppp = ppp
        self.ppp_model = ppp_model
        
        self.cleaners = []
        for outd, imp in cleaners:
            self.cleaners.append(Cleaner(df_train,
                                         df_corrupted,
                                         self.categorical_columns,
                                         self.numerical_columns, 
                                         self.categorical_precision_threshold, 
                                         self.numerical_std_error_threshold,
                                         outlier_detection = outd(df_train,
                                                                  df_corrupted,
                                                                  self.categorical_columns,
                                                                  self.numerical_columns, 
                                                                  self.categorical_precision_threshold, 
                                                                  self.numerical_std_error_threshold),
                                         imputation = imp(df_train,
                                                          df_corrupted,
                                                          self.categorical_columns,
                                                          self.numerical_columns)
                                        )
                                )
            
        
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


    def get_cleaned(self, df_train, df_test, df_corrupted, cols_perturbed):

        print("\nApplying cleaners... \n")
        
        score_no_cleaning = self.ppp.predict_score_ppp(self.ppp_model, df_corrupted)
        print(f"PPP score no cleaning: {score_no_cleaning}")

        summ_clean = {}
        summary_cleaners = []
        
        print(f"PPP scores with cleaning: ")
        cleaning_scores_ppp = []
        for cleaner in self.cleaners:
            df_outliers, df_cleaned = cleaner.apply_cleaner(df_train, df_corrupted)
            outlier_detection_score, imputation_score = cleaner.cleaner_scores(df_test, df_corrupted, df_outliers, df_cleaned, cols_perturbed)
            print(f"\nOutlier detection method: {cleaner.outlier_detection}, Outlier Detection Score: {outlier_detection_score}")
            print(f"Imputation method: {cleaner.imputation}, Imputation Score: {imputation_score}")

            cleaning_score = self.ppp.predict_score_ppp(self.ppp_model, df_cleaned) ## these scores are for the affect of cleaning on the downstream ml models
            
            print(f"{cleaner}: {cleaning_score}")
            cleaning_scores_ppp.append(cleaning_score)

            summ_clean = {
                "Outlier detection method": cleaner.outlier_detection, 
                "Outlier Detection Score": outlier_detection_score, 
                "Imputation method": cleaner.imputation, 
                "Imputation Score": imputation_score, 
                "PPP score with cleaning": cleaning_score
            }
            summary_cleaners.append(summ_clean) ## saving results for returning individuals too
            
        ## finding the best score of the affect of cleaning on the downstream ml models
        roc_scores_for_best = []
        for i in range(len(cleaning_scores_ppp)):
            roc_scores_for_best.append(cleaning_scores_ppp[i]["roc_auc_acore"])

        best_cleaning_idx = pd.Series(roc_scores_for_best).idxmax()
        best_cleaning_score = cleaning_scores_ppp[best_cleaning_idx]

        df_outliers, df_cleaned = self.cleaners[best_cleaning_idx].apply_cleaner(df_train, df_corrupted)
        print(f"\nBest cleaning method:")
        print(f"Cleaning score: {self.cleaners[best_cleaning_idx]}: {best_cleaning_score} \n")

        if best_cleaning_score["roc_auc_acore"] > score_no_cleaning["roc_auc_acore"]:
            print("Cleaning improved the overall score \n\n\n")
        else:
            print("Cleaning didnt't improve the overall score \n\n\n")
            
        return df_outliers, df_cleaned, score_no_cleaning, best_cleaning_score, cleaning_scores_ppp, summary_cleaners
    
    
    def __call__(self, df_train, df_test, df_corrupted, cols_perturbed):
        return self.get_cleaned(df_train, df_test, df_corrupted, cols_perturbed)