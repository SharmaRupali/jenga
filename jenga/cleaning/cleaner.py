from jenga.cleaning.outlier_detection import NoOutlierDetection
from jenga.cleaning.imputation import NoImputation


class Cleaner:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold, outlier_detection=NoOutlierDetection, imputation=NoImputation):
        self.outlier_detection = outlier_detection
        self.imputation = imputation


    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
        
    
    def apply_cleaner(self, df_train, df_corrupted):
        # outliers
        df_outliers, predictors, predictable_cols = self.outlier_detection(df_train, df_corrupted)

        # impute
        df_imputed = self.imputation(df_train, df_outliers, predictors)
        
        return df_imputed