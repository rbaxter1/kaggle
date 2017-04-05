import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
   
def main():
    
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    
    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    #X_train = X_train[:,200:202]
    #X_test = X_test[:,200:202]
    
    #np.ceil(X_train / 10.0) * 10
    
    #X_train= np.ceil(X_train / 10.0) * 10
    #X_test = np.ceil(X_test / 10.0) * 10
    
    # there are 784 dimensions (1 for each pixel)
    # each d has labels 0 to 255
    # one hot encode for each label
    labels = np.arange(0,256,1)
        
    enc_temp = labels.repeat(X_train.shape[1]).reshape((labels.shape[0], X_train.shape[1]))
    #m = np.unique(X_train)
    #enc_temp = np.tile(labels, (labels.shape[0], X_train.shape[1]))#.transpose()
    #enc_temp = np.ones((X_train.shape[1],np.unique(X_train).shape[1])) * [np.unique(X_train) for i in range(X_train.shape[1])]
    #enc_temp = np.ones((X_train.shape[1],256)) * [np.arange(0,256,1) for i in range(X_train.shape[1])]
    #enc_temp = np.transpose(enc_temp)
    enc = OneHotEncoder(dtype=np.int8)
    enc.fit(enc_temp)
    enc.n_values_
    enc.feature_indices_
    X_train_enc = enc.transform(X_train).toarray()
    #X_test_enc = enc.transform(X_test).toarray()

    pipe = Pipeline([('scl', StandardScaler()),
                     ('clf', MLPClassifier(verbose=True))])
    
    print(cross_val_score(pipe, X_train_enc, y_train, cv=10, n_jobs=-1))
    
    #kfold = StratifiedKFold(n_splits=3, shuffle=True).split(X_train_enc, y_train)
    #cv_scores = []
    #test_scores = []
    #for k, (train, test) in enumerate(kfold):
    #    pipe.fit(X_train_enc[train], y_train[train])
    #    cv_score = pipe.score(X_train_enc[test], y_train[test])
    #    cv_scores.append(cv_score)
    #    test_score = pipe.score(X_test_enc, y_test)
    #    test_scores.append(test_score)
    #    print('Fold: %s, Class dist.: %s, CV acc: %.3f, Tets acc: %.3f' % (k+1, np.bincount(y_train[train]), cv_score, test_score))
        
    print('done')
    
if __name__ == '__main__':
    main()
