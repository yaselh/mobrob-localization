# Write script that loads svm file with joblib and saves pickle with lower protocol version
# See https://stackoverflow.com/questions/25843698/valueerror-unsupported-pickle-protocol-3-python2-pickle-can-not-load-the-file

import os
from sklearn.externals import joblib

relative_model_path = '../../../../models/svm/svm12_od.pkl'
model_path = os.path.join(os.path.dirname(__file__), relative_model_path)

if __name__=='__main__':
    model = joblib.load(model_path)
    joblib.dump(model, os.path.splitext(model_path)[0] + '_p2.pkl', protocol=2)
