from abc import abstractmethod

class Imputation:
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columms):
        self.df_train = df_train
        self.df_corrupted = df_corrupted
        
        self.categorical_columns = categorical_columns
        self.numerical_columms = numerical_columms
        
    
    @abstractmethod
    def fit(self, df_train):
        pass
    
    @abstractmethod
    def transform(self, df_corrupted):
        pass
    
    @abstractmethod
    def fit_transform(self, df_train, df_corrupted):
        pass
		
	
	
class MeanModeImputation(Imputation):
    
    def __init__(self, df_train, df_corrupted, categorical_columns, numerical_columms):
        self.means = {}
        self.modes = {}
    
        Imputation.__init__(self, df_train, df_corrupted, categorical_columns, numerical_columms)
    
    
    def fit(self, df_train):
        for col in df_train.columns:
            if col in self.numerical_columms:
                # mean imputer
                mean = np.mean(df_train[col])
                self.means[col] = mean
            elif col in self.categorical_columns:
                # mode imputer
                mode = df_train[col].value_counts().index[0]
                self.modes[col] = mode
                
                
    def transform(self, df_corrupted):
        df_imputed = df_corrupted.copy()
        
        for col in df_corrupted.columns:
            if col in self.numerical_columms:
                # mean imputer
                df_imputed[col].fillna(self.means[col], inplace=True)
            elif col in self.categorical_columns:
                # mode imputer
                df_imputed[col].fillna(self.modes[col], inplace=True)
                
        return df_imputed