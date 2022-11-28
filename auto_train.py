import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

if __name__ == "__main__":
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
    
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))