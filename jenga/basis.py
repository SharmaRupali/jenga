import numpy as np
import tensorflow as tf
import random
import pandas as pd
import openml

from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

# class for loading the datasets, and getting the training and test sets
class Dataset:

    def __init__(self, seed, dataset_name):
        
        ## fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        
        data = openml.datasets.get_dataset(dataset_name)
        
        ## summary
        # print(f"Dataset '{data.name}', target: '{data.default_target_attribute}'")
        # print(data.description[:500])
        print(f"Dataset: {data.name}")
        
        ## load the data
        # X: An array/dataframe where each row represents one example with the corresponding feature values
        # y: the classes for each example
        # categorical_indicator - an array that indicates which feature is categorical
        # attribute_names - the names of the features for the examples(X) and target feature (y)
        X, y, categorical_indicator, self.attribute_names = data.get_data(
            dataset_format='dataframe',
            target=data.default_target_attribute
        )
        
        ## combine the attribute names with the information of them being categorical or not
        # will be used further in order not to manually distinguish between the numerical and categorical features
        self.attribute_types = pd.DataFrame(self.attribute_names, columns=["attribute_names"])
        self.attribute_types['categorical_indicator'] = categorical_indicator
        # print("\nAttribute types: ")
        # display(self.attribute_types)

        self.all_data = X.copy(deep=True)
        self.all_data['class'] = y
        
        ## categorical and numerical columns
        self.categorical_columns = list(self.attribute_types['attribute_names'][self.attribute_types['categorical_indicator'] == True])
        self.numerical_columns = list(self.attribute_types['attribute_names'][self.attribute_types['categorical_indicator'] == False])
        
    
    def get_train_test_data(self):
        
        ''' Get train and test data along with train and test labels.

        Params:
        all_data: dataframe: combined data and labels
        attribute_names: list: names of attributes from the data

        Returns:
        train_data: dataframe:
        train_labels: list
        test_data: dataframe
        test_labels: list
        '''

        train_split, test_split = train_test_split(self.all_data, test_size=0.2)

        train_data = train_split[self.attribute_names]
        train_labels = np.array(train_split['class'])

        test_data = test_split[self.attribute_names]
        test_labels = np.array(test_split['class'])

        return train_data, train_labels, test_data, test_labels



# Base class for binary classification tasks, including training data, test data, a baseline model and scoring
class BinaryClassificationTask(ABC):

    def __init__(self,
                 seed,
                 train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 categorical_columns=None,
                 numerical_columns=None,
                 text_columns=None,
                 is_image_data=False
                 ):

        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = []
        if text_columns is None:
            text_columns = []

        # Fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Train and test data and labels
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.__test_labels = test_labels

        # Information about the data (column types, etc)
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.is_image_data = is_image_data

    # Abstract base method for training a baseline model on the raw data
    @abstractmethod
    def fit_baseline_model(self, train_data, train_labels):
        pass

    # Per default, we compute ROC AUC scores
    def score_on_test_data(self, predicted_label_probabilities):
        return roc_auc_score(self.__test_labels, np.transpose(predicted_label_probabilities)[1])


# Abstract base class for all data corruptions
class DataCorruption:

    # Abstract base method for corruptions, they have to return a corrupted copied of the dataframe
    @abstractmethod
    def transform(self, data):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"