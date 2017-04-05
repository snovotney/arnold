import sklearn
import pandas as pd
import sys
import argparse
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Read in params
parser = argparse.ArgumentParser(description='Train Logistic Regression on text data')
parser.add_argument('text',help='training data in CSV format. One line per text, first value being label')
parser.add_argument('model',help='output file to save model')
args = parser.parse_args()

# load training data
train = pd.read_csv(args.text,header=None,names=['label','text'])

# logistic regression with raw counts and balanced class priors
model = Pipeline([('vect',CountVectorizer()),
                  ('clf',LogisticRegression(class_weight='balanced'))
                 ])

#Cross validation on L2 regularizer
print("Training on {} examples".format(len(train)))
print(train.groupby('label').count())
print("Sweeping across multiple regularization weights...")

parameters = [{'clf__C' : [0.001, 0.01,0.1,0.2,0.5,1]} ]

clf = GridSearchCV(model,parameters)
clf.fit(train['text'],train['label'])

print("Best weight regularizer:{}".format(clf.best_params_['clf__C']))
scores = cross_val_score(clf, train['text'], train['label'], cv=3)
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("Saving model to",args.model)
with open(args.model,'wb') as fh:
    pickle.dump(clf,fh)
    
