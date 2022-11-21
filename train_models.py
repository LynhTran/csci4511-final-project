import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import json
import os

classes = ['not_spam', 'spam']

d1 = pd.read_json(os.path.join('SpamDataset', 'ham_easy.json'))
d2 = pd.read_json(os.path.join('SpamDataset', 'ham_hard.json'))
d3 = pd.read_json(os.path.join('SpamDataset', 'spam.json'))

data = pd.concat([d1, d2, d3])

corpus = data['content']
classes = data['class']

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, classes, test_size=0.20, random_state=10)

#model = MultinomialNB()
model = svm.LinearSVC()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


text = ['''Hi!
Information for you You have a new transfer of funds, you need to withdraw =
them
Further instructions in the attached link:
_______
=F0=9F=91=9B Note!!! You have 5 hours to withdraw your funds after the time
expires, your BTC will be automatically canceled! Withdraw money
''']
v = count_vect.transform(text).toarray()

print(model.predict(v))