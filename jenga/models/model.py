import random
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


class Model:
    
    def __init__(self, seed,
                 train_data, 
                 train_labels, 
                 test_data, 
                 test_labels, 
                 categorical_columns, 
                 numerical_columns,
                 pipeline,
                 learner, 
                 param_grid):
        
        ## fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        ## train and test data and labels
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        
        ## information about the column types in the dataset
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
        ## preprocessing pipeline
        self.pipeline = pipeline
        
        ## information for model parameters
        self.learner = learner
        self.param_grid = param_grid
        
    
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
        

    # method for training a model on the raw data with preprocessing
    def fit_model(self, train_data, train_labels):
        grid_search = GridSearchCV(self.pipeline, self.param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        model = grid_search.fit(train_data, train_labels)

        return model
    
    
    # method for computing ROC AUC scores
    def roc_auc_score_on_test_data(self, model, test_data):
        pred_prob = model.predict_proba(test_data)
        roc_auc_acore = roc_auc_score(self.test_labels, np.transpose(pred_prob)[1])
        
        return roc_auc_acore


    # method for computing evaluation metrics
    def evaluation_metrics(self, model, test_data):
        y_pred = model.predict(test_data)
        
        eval_scores = {
            'roc_auc_acore': roc_auc_score(self.test_labels, np.transpose(model.predict_proba(test_data))[1]),
            # 'accuracy_score': accuracy_score(self.test_labels, y_pred),
            'classification_report': classification_report(self.test_labels, y_pred, output_dict=True)
            # 'precision_score': precision_score(self.test_labels, y_pred),
            # 'recall_score': recall_score(self.test_labels, y_pred),
            # 'f1_score': f1_score(self.test_labels, y_pred),
            
            # 'mean_squared_error': mean_squared_error(self.test_labels, y_pred),
            # 'mean_absolute_error': mean_absolute_error(self.test_labels, y_pred)
        }

        return eval_scores