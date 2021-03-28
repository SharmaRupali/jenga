import numpy as np

from jenga.cleaning.outlier_detection import NoOutlierDetection
from jenga.cleaning.imputation import NoImputation

from sklearn.metrics import classification_report, accuracy_score, mean_squared_error


class Cleaner:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columns, categorical_precision_threshold, numerical_std_error_threshold, outlier_detection=NoOutlierDetection, imputation=NoImputation):
        self.outlier_detection = outlier_detection
        self.imputation = imputation

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns


    def __repr__(self):
        return f"{self.__class__.__name__}: {self.outlier_detection, self.imputation}"
        
    
    def apply_cleaner(self, df_train, df_corrupted):
        # outliers
        df_outliers, predictors = self.outlier_detection(df_train, df_corrupted)

        # impute
        df_imputed = self.imputation(df_train, df_outliers[df_corrupted.columns], predictors)
        
        return df_outliers, df_imputed


    def cleaner_scores(self, df_test, df_corrupted, df_outliers, df_cleaned, cols_perturbed):
        ## y_true is the column value from the original test data
        
        outlier_detection_score = self.outlier_detection_scores(df_test, df_corrupted, df_outliers, cols_perturbed)
        imputation_score = self.imputation_scores(df_test, df_cleaned, cols_perturbed)

        return outlier_detection_score, imputation_score


    def imputation_scores(self, df_test, df_cleaned, cols_perturbed):
        classif_reports = {}

        acc_scores = []
        f1socres = []
        recallscores = []
        precisionscores = []

        mse = []

        for col in cols_perturbed:
            if col in self.categorical_columns:
                classif_reports[col] = classification_report(df_test[col], df_cleaned[col], output_dict=True)

                labels = [k for k in classif_reports[col] if k not in ['accuracy', 'macro avg', 'weighted avg']]

                f1s = []
                res = []
                pres = []

                for label in labels:
                    f1s.append(classif_reports[col][label]['f1-score'])
                    res.append(classif_reports[col][label]['recall'])
                    pres.append(classif_reports[col][label]['precision'])

                f1socres.append(np.mean(f1s))
                recallscores.append(np.mean(res))
                precisionscores.append(np.mean(pres))

                acc_scores.append(accuracy_score(df_test[col], df_cleaned[col]))
            else:
                mse.append(mean_squared_error(df_test[col], df_cleaned[col]))

        imputation_scores_summ = {
            "Precision": np.mean(precisionscores),
            "Recall": np.mean(recallscores),
            "F1-score": np.mean(f1socres),
            "Accuracy": np.mean(acc_scores),
            "Mean Squared Error": np.mean(mse)
        }

        return imputation_scores_summ


    def outlier_detection_scores(self, df_test, df_corrupted, df_outliers, cols_perturbed):
        df_test_out = df_test.copy()

        classif_reports = {}
        acc_scores = []
        f1socres = []
        recallscores = []
        precisionscores = []

        for col in cols_perturbed:
            outiers_man = np.equal(df_test[col], df_corrupted[col])
            outiers_man_ind = outiers_man.index[outiers_man == False]

            non_outliers_man_ind = df_test_out.loc[set(df_test_out.index) - set(outiers_man_ind)].index

            df_test_out.loc[outiers_man_ind, col + "_outlier"] = 1 ## outliers
            df_test_out.loc[non_outliers_man_ind, col + "_outlier"] = 0 ## not outliers

            ## explicitly changing the datatype of the outlier columns: classification_report doesn't take different datatypes
            ## even though the values are just 1s and 0s, they are sometimes read as ints and other times as floats 
            df_test_out[col + "_outlier"] = df_test_out[col + "_outlier"].astype('int')
            df_outliers[col + "_outlier"] = df_outliers[col + "_outlier"].astype('int')

            classif_reports[col] = classification_report(df_test_out[col + "_outlier"], df_outliers[col + "_outlier"], output_dict=True)

            labels = [k for k in classif_reports[col] if k not in ['accuracy', 'macro avg', 'weighted avg']]

            f1s = []
            res = []
            pres = []

            for label in labels:
                f1s.append(classif_reports[col][label]['f1-score'])
                res.append(classif_reports[col][label]['recall'])
                pres.append(classif_reports[col][label]['precision'])

            f1socres.append(np.mean(f1s))
            recallscores.append(np.mean(res))
            precisionscores.append(np.mean(pres))

            acc_scores.append(accuracy_score(df_test_out[col + "_outlier"], df_outliers[col + "_outlier"]))

        od_scores_summ = {
            "Precision": np.mean(precisionscores),
            "Recall": np.mean(recallscores),
            "F1-score": np.mean(f1socres),
            "Accuracy": np.mean(acc_scores)
        }

        return od_scores_summ