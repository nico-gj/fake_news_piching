import pandas as pd
import numpy as np
from models.prediction_models_utils import split_data, create_data_matrices
from read_in_and_cleaning import read_in_and_clean

class Param(object):

    def __init__(self):

        self.features = "tfidf"
        self.text = "body"

        self.df_dict = read_in_and_clean()
        self.data = pd.read_csv("data/{}_{}.csv".format(self.features, self.text))
        print("\n{} observations and {} variables.".format(self.data.shape[0], self.data.shape[1]))
        self.data['label'] = list(self.df_dict['label'].values())

        self.label_name = 'label'

        self.seed = 123

        self.binary_data = (set(self.data[self.label_name])==set([0, 1]))

        self.alpha = 1 # For logistic regressions
        self.max_depth = 5
        self.n_estimators = 5
        self.learning_rate = 0.05

        self.test_perc = 0.3
        self.training_data, self.test_data = split_data(self)

        self.prediction_data = None

        self.training_matrix, self.test_matrix, self.prediction_matrix = create_data_matrices(self)

        self.threshold = 0.5  # Threshold for flagging scores as 1 or 0.
