import numpy as np


# A basic linear Regression model class
# It has only 2 methods: fit and predict
class LinearRegression():

    def __init__(self):
        self.coefficients = []

    def fit(self, features, response):
        x = features
        if type(features) != 'matrix':
            x = np.asmatrix(features)

        y = response
        if type(response) != 'matrix':
            y = np.asmatrix(response).T

        x_t = x.T
        xt_x_inverse = np.dot(x_t, x).I
        self.coefficients = np.dot(xt_x_inverse, np.dot(x_t, y))
        return self

    def predict(self, features):
        x = features
        if type(features) != 'matrix':
            x = np.asmatrix(features)
        return np.asarray(np.dot(x, self.coefficients).T)[0]
