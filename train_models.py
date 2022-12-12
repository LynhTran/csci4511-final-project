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
from pprint import pprint

import json
import os
import time

import warnings
warnings.filterwarnings("ignore")

classes = ['not_spam', 'spam']

print('Reading dataset')

d1 = pd.read_json(os.path.join('SpamDataset', 'apache_ham_easy.json'))
d2 = pd.read_json(os.path.join('SpamDataset', 'apache_ham_hard.json'))
d3 = pd.read_json(os.path.join('SpamDataset', 'apache_spam.json'))
data = pd.concat([d1, d2, d3])

#d1 = pd.read_json(os.path.join('SpamDataset', 'uci_sms_spam.json'))
#d2 = pd.read_json(os.path.join('SpamDataset', 'uci_sms_ham.json'))
#data = pd.concat([d1, d2])

corpus = data['content']
classes = data['class']

print('Performing feature extraction')

count_vect = CountVectorizer(
#    stop_words='english',
    strip_accents='unicode',
    decode_error='replace',
    lowercase=True,
    ngram_range=(1, 1)
)

X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, classes, test_size=0.30, random_state=10)

def train_nb(return_timings=False):
    # Timer starts
    starttime = time.time()

    model = MultinomialNB()
    model.fit(X_train, y_train)

    totaltime = time.time() - starttime
    #print("Training time for Naive Bayes: " + str(totaltime * 1000) + " milliseconds")
    if return_timings:
        return model, totaltime
    else:
        return model


def train_svm(return_timings=False):
    # Timer starts
    starttime = time.time()

    model = svm.LinearSVC()
    model.fit(X_train, y_train)

    totaltime = time.time() - starttime
    #print("Training time for SVM: " + str(totaltime * 1000) + " milliseconds")
    if return_timings:
        return model, totaltime
    else:
        return model

def train_auto(time_limit=3600):
    automl = AutoSklearnClassifier(time_left_for_this_task=time_limit)
    automl.fit(X_train, y_train)
    return automl

def train_auto_v2(time_limit=3600):
    automl = AutoSklearn2Classifier(time_left_for_this_task=time_limit)
    automl.fit(X_train, y_train)
    return automl

def check_model_stats(model):
    classification_timings_avg = 0
    for x in range(0, 10):
        starttime = time.time()
        y_pred = model.predict(X_test)
        totaltime = time.time() - starttime
        classification_timings_avg += totaltime
    classification_timings_avg /= 10
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('F1: %.3f' % f1_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Average Classification time: {}ms'.format(classification_timings_avg * 1000))

nb_model = None
svm_model = None

nb_timings_avg = 0
svm_timings_avg = 0

for x in range(0, 10):
    nb_model, nb_timings = train_nb(return_timings=True)
    svm_model, svm_timings = train_svm(return_timings=True)
    nb_timings_avg += nb_timings
    svm_timings_avg += svm_timings

nb_timings_avg /= 10
svm_timings_avg /= 10

print('Average NB Timing: {}'.format(nb_timings_avg))
print('Average SVM Timings: {}'.format(svm_timings_avg))

print('')

print('NB stats:')
check_model_stats(nb_model)

print('')

print('SVM stats:')
check_model_stats(svm_model)

print('')

print('Training auto-sklearn model v2 for 60sec')
model = train_auto_v2(time_limit=60)
pprint(model.show_models(), indent=4)
check_model_stats(model)

print('')

print('Training auto-sklearn model v2 for 300sec')
model = train_auto_v2(time_limit=300)
pprint(model.show_models(), indent=4)
check_model_stats(model)

print('')

print('Training auto-sklearn model v2 for 600sec')
model = train_auto_v2(time_limit=600)
pprint(model.show_models(), indent=4)
check_model_stats(model)

print('')

print('Training auto-sklearn model v2 for 1800sec')
model = train_auto_v2(time_limit=1800)
pprint(model.show_models(), indent=4)
check_model_stats(model)

print('')

print('Training auto-sklearn model v2 for 3600sec')
model = train_auto_v2(time_limit=3600)
pprint(model.show_models(), indent=4)
check_model_stats(model)