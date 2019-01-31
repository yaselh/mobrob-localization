import numpy as np

from sklearn.base import BaseEstimator

class CupHandleBaseEstimator(BaseEstimator):

    def __init__(self, center_point):
        self.center_point = center_point
        self.best_score = 0

    def fit(self, X, y=None):
        #print('Fit')
        # Fit final estimator with all inliers
        if(X.shape[0] > 1):
            self.point1 = X[0]

        else:
            self.point1 = X[0]
        return self

    def predict(self, X, y=None):
        #print('Predict')
        results = []
        for point in X:
            distance = np.abs(np.linalg.norm(np.cross(self.point1-self.center_point, self.center_point-point))/np.linalg.norm(self.point1-self.center_point))
            results.append(distance)
        results_np = np.array(results)
        results_np = np.reshape(results_np, (results_np.shape[0], 1))
        return results_np


    def score(self, X, y=None):
        #print('Score')
        #score = 1/np.sum(self.results_np)
        score = X.shape[0]
        if score > self.best_score:
            self.best_score = score
        return score
