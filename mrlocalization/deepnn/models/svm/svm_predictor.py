import os
from sklearn.externals import joblib
from svm_train import extract_features
import time

class SVMPredictor:


    relative_model_path = '../../../../models/svm/svm18_od_small.pkl'
    model_path = os.path.join(os.path.dirname(__file__), relative_model_path)

    def initialize_predictor(self):
        self.classifier = joblib.load(SVMPredictor.model_path)

        # Ugly workaround that is not necessary anymore in early 2018 release of scikit-learn
        # see https://github.com/EducationalTestingService/skll/issues/87
        if self.classifier.kernel == u'rbf':
            self.classifier.kernel=str(u'rbf')

    def predict(self, images):
        feature_descriptors = []
        for image in images:
            fd = extract_features(image)
            feature_descriptors.append(fd)
        predicted = self.classifier.predict_proba(feature_descriptors)
        return predicted
