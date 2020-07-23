import random
import numpy as np
from collections import defaultdict

from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import Scaling, GaussianNoise


DEFAULT_CORRUPTIONS = {
    'missing': [MissingValues],
    'categorical': [SwappedValues],
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

        ## removing the corrsponding corruption from the list if not application
        ## UPDATE: check if the corruption type is in the corruptions list and only then remove if appilcable
        if len(self.categorical_columns) == 0:
            if SwappedValues in corruptions:
                print("Can't apply the SwappedValues corruption because there are no categorical columns. \n\n")
                del corruptions[corruptions.index(SwappedValues)]
        elif len(self.numerical_columns) == 0:
            if Scaling in corruptions:
                print("Can't apply the Scaling corruption because there are no numerical columns. \n\n")
                del corruptions[corruptions.index(Scaling)]
            elif GaussianNoise in corruptions:
                print("Can't apply the GaussianNoise corruption because there are no numerical columns. \n\n")
                del corruptions[corruptions.index(GaussianNoise)]

        
        for corruption in corruptions: 
            perturbation, col_perturbed = self.get_perturbation(corruption, fraction)
            print(f"\t... perturbation: {perturbation}")

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
        ## check which type of corruption is given
        perturb_type = ''
        for key, val in DEFAULT_CORRUPTIONS.items():
            for elem in val:
                if elem == corruption:
                    perturb_type = key


        if perturb_type is 'numeric':
            col_to_perturb = random.choice(self.numerical_columns)
            return corruption(col_to_perturb, fraction), [col_to_perturb]
        elif perturb_type is 'categorical':
            col_to_perturb = random.sample(self.categorical_columns, 2)
            return corruption(col_to_perturb[0], col_to_perturb[1], fraction), col_to_perturb
        elif perturb_type is 'missing':
            missigness = random.choice(['MCAR', 'MAR', 'MNAR'])
            col_to_perturb = random.choice(self.numerical_columns + self.categorical_columns)
            na_value = np.nan
            return corruption(col_to_perturb, fraction, na_value, missigness), [col_to_perturb]