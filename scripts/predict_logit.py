import sklearn
import pandas as pd
import numpy as np
import sys
import argparse
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Predict labels on test data using logistic regression')
parser.add_argument('text',help='test data. One line per text, first value being label')
parser.add_argument('pred',help='Prediction file to write out most likely hypothesis')
parser.add_argument('model',help='file containing logit model')
args = parser.parse_args()

test = pd.read_csv(args.text,header=None,names=['label','text'])
with open(args.model,'rb') as fh:
    clf = pickle.load(fh)
    
test_hyp = clf.predict(test['text'])
with open(args.pred, 'w') as fh:
    for label in test_hyp:
        fh.write(label + "\n")

if len(test['label'].unique()) > 1:
    print("Cross Validation Logistic Regression Test Accuracy: {:.2f}".format(np.mean(test_hyp == test['label'])))
    print(classification_report(test['label'],test_hyp))
else:
    print("All labels are the same, not computing accuracy")
    
print("Predictions written to",args.pred)
