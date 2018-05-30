import pickle
import xgboost
import numpy as np
import pandas as pd
from Encoder import XGBoostEncodingGenerator

class XGB:
    """XGBoost

    API wrapper for trained XGBoost model
    """

    def __init__(self, path='../models/english/en-xgb[0.5]'):
        """Initialize & load model with its parameters"""
        # init model
        self.model = pickle.load(open(path, "rb"))
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
        enc_gen = XGBoostEncodingGenerator()
        
        contextual_data = np.array(enc_gen.context_window_transform(encoded_data, self.pad_size))
        columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
       '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
       '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
       '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
       '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
       '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84',
       '85', '86', '87', '88', '89', '90', '91', '92', '93']
        X = pd.DataFrame(data=contextual_data, columns=columns)

        # classify as RemainSelf or ToBeNormalized
        y = self.model.predict(X)
        y_labels = [self.labels[int(i)] for i in y]
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
