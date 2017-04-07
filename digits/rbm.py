import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer, OneHotEncoder

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def rbm():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    
    # n_components=256, learning_rate=0.1, batch_size=10, n_iter=10    
    clf = Pipeline([('rbm', BernoulliRBM()), ('lr', LogisticRegression())])

    param_grid = {"rbm__n_components": np.arange(1,500,50),
                  "rbm__learning_rate": np.arange(0.01,0.1,0.01),
                  "lr__C": np.arange(1,1000,100)}
    gs = GridSearchCV(clf, param_grid=param_grid,n_jobs=5)
    st = time()
    gs.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - st, len(gs.cv_results_['params'])))
    report(gs.cv_results_)
    
def testRBM():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    #sc = StandardScaler()
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_test_std = sc.transform(X_test)
    
    #X_train_std = X_train
    #X_test_std = X_test
    
    train_scores = []
    test_scores = []
    #rbm = BernoulliRBM(n_components=X_train.shape[1])
    #for n_components in np.arange(1, X_train_std.shape[1]):
    for threshold in np.arange(0, 300, 50):
        #rbm = BernoulliRBM(n_components=n_components, verbose=True
        X_train_bin = Binarizer(threshold=threshold).fit_transform(X_train)
        rbm = BernoulliRBM(verbose=True)
        rbm.fit(X_train_bin, y_train)
        
        X_train_transform = rbm.transform(X_train_bin)
        
        #clf = MLPClassifier(verbose=True)
        #clf = SGDClassifier(loss='log', verbose=True)
        clf = LogisticRegression(verbose=True)
        #clf.coef_ = rbm.components_[0,:]
        clf.fit(X_train_transform, y_train)
        
        train_score = clf.score(X_train_transform, y_train)
        train_scores.append(train_score)
        
        print(threshold)
        print(train_score)
        
        X_test_bin = Binarizer(threshold=threshold).fit_transform(X_test)
        X_test_transform = rbm.transform(X_test_bin)
        test_score = clf.score(X_test_transform, y_test)
        test_scores.append(test_score)
        print(test_score)
        

    
    filename = 'vc_rbm_thresh.png'
    title = "Learning Curve: RBM threshold"
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    plt.grid()
    
    plt.plot(x_axis, train_scores, color="r",
             label="Training score")
    plt.plot(x_axis, test_scores, color="g",
             label="Testing score")
    
    plt.legend(loc="best")
    plt.savefig(filename)
                
    print('done')
    
    
def testRBM2():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # there are 784 dimensions (1 for each pixel)
    # each d has labels 0 to 255
    # one hot encode for each label
    labels = np.arange(0,256,1)
    enc_temp = labels.repeat(X_train.shape[1]).reshape((labels.shape[0], X_train.shape[1]))
    enc = OneHotEncoder(dtype=np.int8)
    enc.fit(enc_temp)
    #enc.n_values_
    #enc.feature_indices_
    X_train_ohe = enc.transform(X_train).toarray()
    
    rbm = BernoulliRBM(verbose=True)
    rbm.fit(X_train_ohe, y_train)
    X_train_rbm = rbm.transform(X_train_ohe)
    
    clf = LogisticRegression(verbose=True)
    clf.fit(X_train_rbm, y_train)
    train_score = clf.score(X_train_rbm, y_train)
    print(train_score)
        
    X_test_rbm = rbm.transform(X_test_enc)
    test_score = clf.score(X_test_rbm, y_test)
    print(test_score)
    
    print('done')
    
if __name__ == '__main__':
    testRBM2()
