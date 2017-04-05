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
   
   
def test_enc():
    X_train = np.ones((3,2)) * [[  0,   2], [  1,   3], [  2,   4]]
    
    labels = np.arange(0,5,1)
    enc_temp = labels.repeat(X_train.shape[1]).reshape((labels.shape[0], X_train.shape[1]))
    enc = OneHotEncoder(dtype=np.int8)
    enc.fit(enc_temp)
    #enc.n_values_
    #enc.feature_indices_
    X_train_enc = enc.transform(X_train).toarray().transpose()
    
    
    
def main():
    
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    
    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)
    
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
    enc = OneHotEncoder(dtype=np.int8)
    enc.fit(enc_temp)
    #enc.n_values_
    #enc.feature_indices_
    X_train_enc = enc.transform(X_train).toarray()
    
    # question-can i eliminate any rows where the sum is 0? i wonder
    #X_train_enc.sum(0)
    
    Cs = [100, 75, 50, 25, 1]
    solvers = ['newton-cg', 'lbfgs', 'sag']
    #solvers = ['liblinear']
    all_scores = []
    for solver in solvers:
        for C in Cs:
            # have to split up the training because of memory limitations
            clf = LogisticRegression(verbose=True, C=C, solver=solver, warm_start=True, n_jobs=-1)
            index = np.arange(1, X_train_enc.shape[0], 1)
            splits = np.array_split(index, 10)
            for i in range(len(splits)):
                clf.fit(X_train_enc[splits[i]], y_train[splits[i]])        
            
            # free memory
            #X_train_enc = None
            
            X_test_enc = enc.transform(X_test).toarray()
            print("scoring: ")
            score = clf.score(X_test_enc, y_test)
            print(score)
            
            all_scores.append((C, solver, score))
            
            with open(r'test.txt', 'w') as f:
                f.write(" ".join(map(str, all_scores)))
    
    #print(cross_val_score(clf, X_train_enc, y_train, n_jobs=-1))
    
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
    #test_enc()
    main()
