import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

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

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, classes, test_size=0.30, random_state=1)

nb = MultinomialNB()
nb.fit(X_train, y_train)

print("Accuracy of Model",nb.score(X_test,y_test)*100,"%")