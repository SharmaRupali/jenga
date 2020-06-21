from jenga.cleaning.outlier_detection import NoOutlierDetection
from jenga.cleaning.imputation import NoImputation


class Cleaner:
    
    def __init__(self, 
                 df_train,
                 df_corrupted,
                 categorical_columns,
                 numerical_columns,
                 outlier_detection=NoOutlierDetection, 
                 imputation=NoImputation):
        self.outlier_detection = outlier_detection
        self.imputation = imputation
        
    
    def apply_cleaner(self, df_train, df_corrupted, categorical_columns, numerical_columns):
        df_cleaned = self.outlier_detection(df_train, df_corrupted, 
                                            categorical_columns, 
                                            numerical_columns).fit_transform(df_train, df_corrupted)
        
        # do something for fixing/removing the outliers
        if 'outlier' in df_cleaned.columns:
            ### TODO 
            df_cleaned = df_cleaned.drop('outlier', axis=1)
            
        # impute
        df_cleaned = self.imputation(df_train, df_cleaned, 
                                     categorical_columns, 
                                     numerical_columns).fit_transform(df_train, df_corrupted)
        
        return df_cleaned