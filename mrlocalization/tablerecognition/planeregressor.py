import numpy as np

from math import isclose

from sklearn.base import BaseEstimator
from sklearn.linear_model import RANSACRegressor
from sklearn.utils.estimator_checks import check_estimator

residual_threshold = 0.02

class PlaneBaseEstimator(BaseEstimator):


    def fit(self, X, y=None):
        #print('Fit')
        # Fit final estimator with all inliers
        if X.shape[0] > 3:
            point0 = X[0]
            point1 = X[int(0.25 * X.shape[0])]
            point2 = X[int(0.5 * X.shape[0])]
        else:
            point0 = X[0]
            point1 = X[1]
            point2 = X[2]
        p0_to_p1 = point1 - point0
        p0_to_p2 = point2 - point0
        normal_vector = np.cross(p0_to_p1, p0_to_p2)
        #normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.point0 = point0
        self.a = normal_vector[0]
        self.b = normal_vector[1]
        self.c = normal_vector[2]
        self.d = -(self.a * point0[0] + self.b * point0[1] + self.c * point0[2])
        if X.shape[0] > 3:
            self.inliers = X
        else:
            self.inliers = None
        return self

    def predict(self, X, y=None):
        #print('Predict')
        results = []
        normal_vector = np.array([self.a, self.b, self.c])
        for point in X:
            result = np.abs(self.a * point[0] + self.b * point[1] + self.c * point[2] + self.d) / np.linalg.norm(normal_vector)
            results.append(result)
        results_np = np.array(results)
        results_np = np.reshape(results_np, (results_np.shape[0], 1))
        self.prediction_result = results_np
        return results_np


    def score(self, X, y=None):
        #print('Score')
        return X.shape[0]




class PlaneRegressor:

    def fit(self, pointcloud_np):
        y = np.zeros((pointcloud_np.shape[0], 1))
        ransac = RANSACRegressor(base_estimator=PlaneBaseEstimator(), min_samples=3, max_trials=500, residual_threshold=residual_threshold, loss='absolute_loss', max_skips = 500)
        try:
            ransac.fit(pointcloud_np[:,:3], y)
            best_model = ransac.estimator_
        except:
            best_model = None
            print('Some error occured while fitting!')
        if best_model:
            normal_vector = np.array([best_model.a, best_model.b, best_model.c])
            normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
            model = np.array(normal_vector_normalized)
            np.append(model, best_model.d)
            return (model, best_model.inliers)
        else:
            return None

if __name__=='__main__':
    print('Running plane regressor...')
    check_estimator(PlaneBaseEstimator)
