import random
import numpy as np
from collections import defaultdict

from jenga.corruptions.generic import MissingValues, SwappedValues, CategoricalShift
from jenga.corruptions.numerical import Scaling, GaussianNoise


DEFAULT_CORRUPTIONS = {
    'missing': [MissingValues],
    'categorical': [SwappedValues, CategoricalShift],
    'numeric': [Scaling, GaussianNoise]
}


class Perturbation:
    
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns


    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
        
    
    def apply_perturbation(self, df, corruptions, fraction):
        df_corrupted = df.copy()
    
        perturbations = []
        cols_perturbed = []

        summary_col_corrupt = defaultdict(list)

        for corruption in corruptions: 
            if (len(self.categorical_columns) == 0) & (corruption == CategoricalShift):
                print("No categorical columns to apply CategoricalShift!")
                continue;
            elif (len(self.numerical_columns) == 0) & (corruption in [Scaling, GaussianNoise]):
                print(f'No numeric columns to apply {corruption}')
                continue;
            else:
                perturbation, col_perturbed = self.get_perturbation(corruption, fraction)

                print(f"\tperturbation: {perturbation}")

                summary_col_corrupt[tuple(col_perturbed)].append(perturbation) ## saving results for returning individuals too

                ## storing for conservation
                # maybe we want to apply the same set of perturbations again: useful for the CleanML scenarios
                # or maybe we want to reuse the columns that were perturbed
                perturbations.append(perturbation) 
                cols_perturbed.append(col_perturbed)

                df_corrupted = perturbation.transform(df_corrupted)

        ## cols_perturbed is a list of lists, flattening it here
        cols_perturbed = [col for sublist in cols_perturbed for col in sublist]

        return df_corrupted, perturbations, cols_perturbed, summary_col_corrupt


    def get_perturbation(self, corruption, fraction):
        missingness = random.choice(['MCAR', 'MAR', 'MNAR'])
        sampling = missingness

        col_to_perturb = ''
        if corruption in [Scaling, GaussianNoise]:
            col_to_perturb = random.choice(self.numerical_columns)
        elif corruption == CategoricalShift:
            col_to_perturb = random.choice(self.categorical_columns)
        else:
            col_to_perturb = random.choice(self.numerical_columns + self.categorical_columns)

        if corruption == MissingValues:
            return corruption(column=col_to_perturb, fraction=fraction, missingness=missingness), [col_to_perturb]
        else:
            return corruption(column=col_to_perturb, fraction=fraction, sampling=missingness), [col_to_perturb]