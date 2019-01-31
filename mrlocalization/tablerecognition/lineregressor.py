import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import RANSACRegressor

class LineBaseEstimator(BaseEstimator):


    def fit(self, X, y=None):
        #print('Fit')
        # Fit final estimator with all inliers
        if(X.shape[0] > 2):
            point0 = X[0]
            point1 = X[X.shape[0] - 1]
            # vec = np.copy(point0) # copy
            # for i in range(0, X.shape[0]):
            #     vec += (X[i] - point0)
            # vec /= (X.shape[0]-1)
            # vec /= np.linalg.norm(vec)
            # point1 = point0 + vec
        else:
            point0 = X[0]
            point1 = X[1]
        p0_to_p1 = point1 - point0
        self.point0 = point0
        self.point1 = point1
        self.p0_to_p1 = p0_to_p1
        self.p0_to_p1 = self.p0_to_p1 / np.linalg.norm(self.p0_to_p1)
        return self

    def predict(self, X, y=None):
        #print('Predict')
        results = []
        for point in X:
            result = np.linalg.norm(np.cross(self.point1-self.point0, self.point0-point))/np.linalg.norm(self.point1-self.point0)
            results.append(result)
        results_np = np.array(results)
        results_np = np.reshape(results_np, (results_np.shape[0], 1))
        self.results_np = results_np
        return results_np


    def score(self, X, y=None):
        #print('Score')
        #score = 1/np.sum(self.results_np)
        score = X.shape[0]
        return score


class LineRegressor:


    def fit(self, pointcloud_np):
        y = np.zeros((pointcloud_np.shape[0], 1))
        ransac = RANSACRegressor(base_estimator=LineBaseEstimator(), min_samples=2, max_trials=500, residual_threshold=0.01, loss='absolute_loss', max_skips = 500)
        ransac.fit(pointcloud_np[:,:3], y)
        try:
            ransac.fit(pointcloud_np[:,:3], y)
            best_model = ransac.estimator_
        except:
            best_model = None
            print('Some error occured while fitting!')
        if best_model:
            model = np.array([best_model.point0, best_model.point1, best_model.p0_to_p1])
            return model
        else:
            return None
