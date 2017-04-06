from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


train = pd.read_csv("train.csv")
y = train.values[:,0]
X = train.values[:,1:]

# split the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)

# there are 784 dimensions (1 for each pixel)
# each d has labels 0 to 255
# one hot encode for each label
labels = np.arange(0,256,1)
enc_temp = labels.repeat(X_train.shape[1]).reshape((labels.shape[0], X_train.shape[1]))
enc = OneHotEncoder(dtype=np.int8)
enc.fit(enc_temp)
X_train = enc.transform(X_train).toarray()
X_test = enc.transform(X_test).toarray()

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()