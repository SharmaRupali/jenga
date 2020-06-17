## good
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from jenga.basis import BinaryClassificationTask


class SimpleModel(BinaryClassificationTask):
    
    
    def __init__(self, seed, train_data, train_labels, test_data, test_labels, attribute_types, learner, param_grid):
        
        self.categorical_columns = list(attribute_types['attribute_names'][attribute_types['categorical_indicator'] == True])
        self.numerical_columms = list(attribute_types['attribute_names'][attribute_types['categorical_indicator'] == False])
        
        BinaryClassificationTask.__init__(self, seed, train_data, train_labels, test_data, test_labels, self.categorical_columns, self.numerical_columms)
        
        self.learner = learner
        self.param_grid = param_grid
        
    
    def fit_baseline_model(self, train_data, train_labels):
        
        ''' Get a trained model.
    
        Params:
        train_data: dataframe
        train_labels: list
        learner: estimator object: estimator to be used
        param_grid: dict: param names as keys and lists of param settings to try as values

        Returns:
        categorical_columns
        numerical_columms
        model
        '''
        
        # preprocessing pipeline for numerical columns
        transformer_numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('standard_scale', StandardScaler())
        ])

        # preprocessing pipeline for categorical columns
        transformer_categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__NA__')),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
        ])

        # preprocessor
        feature_transform = ColumnTransformer(transformers=[
            ('categorical_features', transformer_categorical, self.categorical_columns),
            ('numerical_features', transformer_numeric, self.numerical_columms)
        ])

        ## prediction pipeline: append classifier (learner) to the preprocessing pipeline
        pipeline = Pipeline([
            ('features', feature_transform),
            ('learner', self.learner)
        ])

        grid_search = GridSearchCV(pipeline, self.param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        model = grid_search.fit(train_data, train_labels)

        return model