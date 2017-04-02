import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from time import time

def plot_series(self, x, y, y_std, y_lab, colors, markers, title, xlab, ylab, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i in range(len(y)):
        plt.plot(x, y[i],
                     color=colors[i], marker=markers[i],
                     markersize=5,
                     label=y_lab[i])

        if None != y_std[i]:
            plt.fill_between(x,
                                 y[i] + y_std[i],
                                 y[i] - y_std[i],
                                 alpha=0.15, color=colors[i])

    plt.grid()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')

    plt.savefig(filename)



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
            
def rf():
    test = pd.read_csv("test.csv")
    X_test = test.values
    
    train = pd.read_csv("train.csv")
    y_train = train.values[:,0]
    X_train = train.values[:,1:]
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    clf = RandomForestClassifier()
    
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 4, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X_train_std, y_train)    
    
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train_std, y_train)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train_std[train], y_train[train])
        score = clf.score(X_train_std[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))



def qda():
    test = pd.read_csv("test.csv")
    X_test = test.values
    
    train = pd.read_csv("train.csv")
    y_train = train.values[:,0]
    X_train = train.values[:,1:]
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    clf = QuadraticDiscriminantAnalysis()
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train_std, y_train)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train_std[train], y_train[train])
        score = clf.score(X_train_std[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

    
    
def nn():
    test = pd.read_csv("test.csv")
    X_test = test.values
    
    train = pd.read_csv("train.csv")
    y_train = train.values[:,0]
    X_train = train.values[:,1:]
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    #pipe = Pipeline([('scl', StandardScaler()),
    #                 ('clf', MLPClassifier(hidden_layer_sizes=(300,300,300)))])
    
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(200,200))
    '''
    param_grid = {"hidden_layer_sizes": [(100,),
                                         (100,100),
                                         (100,100,100),
                                         (200,),
                                         (200,200),
                                         (200,200,200),
                                         (300,),
                                         (300,300),
                                         (300,300,300),
                                         (400,),
                                         (400,400),
                                         (400,400,400)]}

    
    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
    start = time()
    grid_search.fit(X_train_std, y_train)    

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    
    
    Model with rank: 1
    Mean validation score: 0.966 (std: 0.000)
    Parameters: {'hidden_layer_sizes': (200, 200)}
    
    Model with rank: 2
    Mean validation score: 0.966 (std: 0.001)
    Parameters: {'hidden_layer_sizes': (400, 400)}
    
    Model with rank: 3
    Mean validation score: 0.966 (std: 0.001)
    Parameters: {'hidden_layer_sizes': (300, 300)}
    '''
    kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train_std, y_train)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train_std[train], y_train[train])
        score = clf.score(X_train_std[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
        
    #clf.fit(X_train_std, y_train)
    #pred = clf.predict(X_test_std)
    
    #kaggle = np.column_stack((np.arange(1, pred.shape[0]+1, 1), pred))
    #df = pd.DataFrame(kaggle)
    #df.columns = ['ImageId','Label']
    
    
    print('ready')
    
    #ImageId,Label
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1).split(y_train, y_train)    
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe.fit(X_train[train], y_train[train])
        score = pipe.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,
              np.bincount(y_train[train]), score))

    print('done')
    
def main():
    #test = pd.read_csv("test.csv")
    #train_01 = test.values[np.logical_or(test.values[:,0]==0, test.values[:,0]==1)]
    
    #X_test = test.values
    
    train = pd.read_csv("train.csv")
    train_01 = train.values[np.logical_or(train.values[:,0]==0, train.values[:,0]==1)]
    y_train_01 = train_01[:,0]
    X_train_01 = train_01[:,1:]
    #y_train = train.values[:,0]
    #X_train = train.values[:,1:]
    
    pipe = Pipeline([('scl', StandardScaler()),
                     ('clf', LogisticRegression(solver='sag', n_jobs=-1))])
    
    
    
    clf = LogisticRegression(solver='sag', n_jobs=-1)
    #clf.fit(X_train_01, y_train_01)
    
    #pred = clf.predict()
    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(y_train_01, y_train_01)
        
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train_01[train], y_train_01[train])
        score = clf.score(X_train_01[test], y_train_01[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,
              np.bincount(y_train_01[train]), score))
            
    #sc = StandardScaler()
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_test_std = sc.transform(X_test)
    
    #clf = LogisticRegression()
    #clf.fit(X_train_std, y_train)
    
    print('done')
    


if __name__ == '__main__':
    nn()