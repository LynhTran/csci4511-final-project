# Apache SpamAssassin email dataset: https://spamassassin.apache.org/old/publiccorpus/readme.html
# Enron-Spam email dataset: http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/readme.txt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import json
import os

import time

from pprint import pprint

classes = ['not_spam', 'spam']

print('Reading dataset')

d1 = pd.read_json(os.path.join('SpamDataset', 'apache_ham_easy.json'))
d2 = pd.read_json(os.path.join('SpamDataset', 'apache_ham_hard.json'))
d3 = pd.read_json(os.path.join('SpamDataset', 'apache_spam.json'))

data = pd.concat([d1, d2, d3])

corpus = data['content']
classes = data['class']

print('Performing feature extraction')

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, classes, test_size=0.20, random_state=10)

def train_nb():
    # Timer starts
    starttime = time.time()

    model = MultinomialNB()
    model.fit(X_train, y_train)

    totaltime = time.time() - starttime
    print("Training time for Naive Bayes: " + str(totaltime * 1000) + " milliseconds")
    return model

def train_svm():
    # Timer starts
    starttime = time.time()

    model = svm.LinearSVC()
    model.fit(X_train, y_train)

    totaltime = time.time() - starttime
    print("Training time for SVM: " + str(totaltime * 1000) + " milliseconds")
    return model

def train_auto():
    automl = AutoSklearnClassifier(time_left_for_this_task=600)
    automl.fit(X_train, y_train)
    return automl

def train_auto_v2():
    automl = AutoSklearn2Classifier(time_left_for_this_task=600)
    automl.fit(X_train, y_train)
    return automl

def check_model_stats(model):
    y_pred = model.predict(X_test)
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('F1: %.3f' % f1_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

print('Training model: Naive Bayes')
model = train_nb()

print('Training model: SVM')
model = train_svm()

# print('Training model')
# model = train_auto()

# check_model_stats(model)
# pprint(model.show_models(), indent=4)

# print('Training model v2')
# model = train_auto_v2()

# check_model_stats(model)
# pprint(model.show_models(), indent=4)