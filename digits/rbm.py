import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 

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

    param_grid = {"rbm__n_components": np.arange(1,125,25)}
    gs = GridSearchCV(clf, param_grid=param_grid)
    st = time()
    gs.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - st, len(gs.cv_results_['params'])))
    report(gs.cv_results_)
    
if __name__ == '__main__':
    rbm()
