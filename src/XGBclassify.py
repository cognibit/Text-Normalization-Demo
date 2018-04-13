import xgboost
import numpy as np
import pandas as pd


class XGB:
    """XGBoost

    API wrapper for trained XGBoost model
    """

    def __init__(self, path='../models/english/en-xgb[0.5]'):
        """Initialize & load model with its parameters"""
        # init model
        self.model = xgboost.Booster({'nthread': 4}, model_file = path)
        # load model
        # self.model.load_model(path)
        # init parameters
        self.max_num_features = 30
        self.pad_size = 1
        self.boundary_letter = -1
        self.space_letter = 0
        self.labels = ['RemainSelf', 'ToBeNormalized']
        return None

    def predict(self, data):
        """XGBoost prediction

        Classifies the dataframe's 'before' tokens

        Args:
            data: pandas dataframe having 'before' column

        Returns:
            y_labels: list of class labels
        """
        # pre-process data
        encoded_data = self._encode(data)
        contextual_data = self._context_window_transform(encoded_data, self.pad_size)
        X = xgboost.DMatrix(np.array(contextual_data))
        # classify as RemainSelf or ToBeNormalized
        y = self.model.predict(X)
        y_labels = [self.labels[int(i)] for i in y]
        # # append a class column in the dataframe
        # # which holds the classification labels
        # data['class'] = y_labels
        return y_labels

    def _encode(self, data):
        """Encodes data into vectors"""
        encoded_data = []
        for x in data['before'].values:
            x_row = np.ones(self.max_num_features, dtype=int) * self.space_letter
            for xi, i in zip(list(str(x)), np.arange(self.max_num_features)):
                x_row[i] = ord(xi)
            encoded_data.append(x_row)
        return encoded_data

    def _context_window_transform(self, data, pad_size):
        """Transforms into a context window"""
        pre = np.zeros(self.max_num_features)
        pre = [pre for x in np.arange(pad_size)]
        data = pre + data + pre
        context_data = []
        for i in np.arange(len(data) - pad_size * 2):
            row = []
            for x in data[i: i + pad_size * 2 + 1]:
                row.append([self.boundary_letter])
                row.append(x)
            row.append([self.boundary_letter])
            context_data.append([int(x) for y in row for x in y])
        return context_data
