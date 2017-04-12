import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, RobustScaler, Binarizer
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



def report(results, n_top=20):
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
    report(grid_search.cv_results_, 10)


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

    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]

    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)

    pipe = Pipeline([('scl', StandardScaler()),
                     ('clf', MLPClassifier(verbose=True))])


    #pipe = Pipeline([('scl', StandardScaler()),
    #                 ('clf', MLPClassifier(verbose=True, tol=1e-6, hidden_layer_sizes=(1800,900,400)))])

    #pipe = Pipeline([('scl', StandardScaler()),
    #                 ('clf', SVC(verbose=True))])

    #pipe = Pipeline([('rbm', BernoulliRBM(learning_rate = 0.06, n_iter = 20, n_components = 100, verbose=True)),
    #                 ('logistic', LogisticRegression(C = 6000.0))])

    #clf = MLPClassifier(tol=1e-6, verbose=True, warm_start=True, early_stopping=True, validation_fraction=0.1, hidden_layer_sizes=(300,300,300))
    #clf = MLPClassifier(hidden_layer_sizes=(300,300,300), warm_start=True, verbose=True)

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


    param_grid = {"hidden_layer_sizes": [(100,),
                                         (100,100)]}
    '''     

    '''
    Model with rank: 1
    Mean validation score: 0.959 (std: 0.001)
    Parameters: {'clf__hidden_layer_sizes': (250,)}

    Model with rank: 2
    Mean validation score: 0.957 (std: 0.001)
    Parameters: {'clf__hidden_layer_sizes': (200,)}

    Model with rank: 3
    Mean validation score: 0.957 (std: 0.001)
    Parameters: {'clf__hidden_layer_sizes': (150,)}

    GridSearchCV took 394.77 seconds for 4 candidate parameter settings.
    Model with rank: 1
    Mean validation score: 0.958 (std: 0.002)
    Parameters: {'clf__hidden_layer_sizes': (250, 200)}

    Model with rank: 2
    Mean validation score: 0.957 (std: 0.002)
    Parameters: {'clf__hidden_layer_sizes': (250, 150)}

    Model with rank: 3
    Mean validation score: 0.957 (std: 0.003)
    Parameters: {'clf__hidden_layer_sizes': (250, 250)}

    GridSearchCV took 344.26 seconds for 4 candidate parameter settings.
    Model with rank: 1
    Mean validation score: 0.961 (std: 0.001)
    Parameters: {'clf__hidden_layer_sizes': (250, 200, 250)}

    Model with rank: 2
    Mean validation score: 0.960 (std: 0.001)
    Parameters: {'clf__hidden_layer_sizes': (250, 200, 200)}

    Model with rank: 3
    Mean validation score: 0.960 (std: 0.002)
    Parameters: {'clf__hidden_layer_sizes': (250, 200, 100)}

    '''

    #hl = [(i,) for i in np.arange(100, 300, 50)]
    #hl = [(250,200,i) for i in np.arange(100, 300, 50)]
    #param_grid = {"clf__hidden_layer_sizes": hl}
    #grid_search = GridSearchCV(pipe, param_grid=param_grid, verbose=True, n_jobs=-1)
    #start = time()
    #grid_search.fit(X_train, y_train)    

    #print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #      % (time() - start, len(grid_search.cv_results_['params'])))
    #report(grid_search.cv_results_)

    '''
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

    #enc = OneHotEncoder()
    #enc.fit(X_train)
    #enc.n_values_
    #enc.feature_indices_
    #enc.transform(X_train).toarray()

    #kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train, y_train)
    #scores = []
    #for k, (train, test) in enumerate(kfold):
    #    pipe.fit(X_train[train], y_train[train])
    #    score = pipe.score(X_train[test], y_train[test])
    #    scores.append(score)
    #    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

    #pipe = Pipeline([('scl', StandardScaler()),
    #                 ('clf', MLPClassifier(verbose=True, tol=1e-8, max_iter=63))])
    #pipe.fit(X_train, y_train)

    #pred = pipe.predict(X_test)
    #score = pipe.score(X_test, y_test)

    #hl = [(800,i) for i in np.arange(100, 900, 100)]
    hl = [(i,) for i in np.arange(100, 2000, 100)]
    train_scores, test_scores = validation_curve(pipe, X_train, y_train, "clf__hidden_layer_sizes", hl, verbose=True, n_jobs=-1)
    train_sizes = np.arange(100, 2000, 100)
    title = "Validation Curve: Hidden Layers"
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()    
    plt.savefig("vc1.png")


    lri = [np.arange(0.0001, 0.1, 0.005)]
    train_scores, test_scores = validation_curve(pipe, X_train, y_train, "clf__learning_rate_init", lri, verbose=True, n_jobs=-1)

    train_sizes = np.arange(0.0001, 0.1, 0.005)
    title = "Validation Curve: Learning Rate Init"
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("Learning Rate Init")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()    
    plt.savefig("vc_alpha.png")

    # validate against test
    '''
    train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, train_sizes=np.arange(0.25, 1.25, 0.25))

    title = "Learning Curve"
    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    '''




    #clf.fit(X_train_std, y_train)
    #test = pd.read_csv("test.csv")
    #X_test = test.values    
    #X_test_std = sc.transform(X_test)

    #pred = clf.predict(X_test_std)

    #kaggle = np.column_stack((np.arange(1, pred.shape[0]+1, 1), pred))
    #df = pd.DataFrame(kaggle)
    #df.columns = ['ImageId','Label']
    #df.to_csv("digits.csv", index=False)

    #ImageId,Label
    print('done')



def nn2():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]

    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)


    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)

    sc.fit(X_test)
    X_test_std = sc.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(300,300,300), warm_start=True, verbose=True)

    kfold = StratifiedKFold(n_splits=10, shuffle=True).split(X_train_std, y_train)

    test_scores = []
    scores = []
    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train_std[train], y_train[train])
        score = clf.score(X_train_std[test], y_train[test])
        scores.append(score)

        test_score = clf.score(X_test_std, y_test)
        test_scores.append(test_score)

        print('Fold: %s, Class dist.: %s, Acc: %.3f, Test: %.3f' % (k+1, np.bincount(y_train[train]), score, test_score))

    #clf.fit(X_train_std, y_train)
    #test = pd.read_csv("test.csv")
    #X_test = test.values    
    #X_test_std = sc.transform(X_test)

    #pred = clf.predict(X_test_std)

    #kaggle = np.column_stack((np.arange(1, pred.shape[0]+1, 1), pred))
    #df = pd.DataFrame(kaggle)
    #df.columns = ['ImageId','Label']
    #df.to_csv("digits.csv", index=False)

    #ImageId,Label
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


def plothist():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]
    X1d = X.flatten()
    
    plt.hist(X1d)
    plt.title("Pixel Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
        
    

def nn3():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]

    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)
    
    threshold = 224
    X_train_std = Binarizer(threshold=threshold).fit_transform(X_train)
    X_test_std = Binarizer(threshold=threshold).fit_transform(X_test)
    #sc = StandardScaler()
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_train_std = X_train

    #sc.fit(X_test)
    #X_test_std = sc.transform(X_test)
    #X_test_std = X_test
    
    c = 0.001
    
    for n_components in np.linspace(300, 800, 5):
        rbm = BernoulliRBM(learning_rate=c, verbose=True)
        rbm.fit(X_train_std, y_train)
        X_train_std = rbm.transform(X_train_std)
        X_test_std = rbm.transform(X_test_std)
        
        activations = ['identity', 'logistic', 'tanh']
        activations = ['logistic']
        for activation in activations:
            #clf = LogisticRegression(max_iter=1, C=c, multi_class='multinomial', solver='sag', warm_start=True, verbose=True)
            clf = MLPClassifier(activation=activation, max_iter=1, warm_start=True, verbose=True)
    
            test_scores = []
            train_scores = []
            #clf.fit(X_train_std, y_train)
    
            x_axis = range(1000)
            for k in x_axis:
                clf.fit(X_train_std, y_train)
                train_score = clf.score(X_train_std, y_train)
                train_scores.append(train_score)
                test_score = clf.score(X_test_std, y_test)
                test_scores.append(test_score)
                print('Iteration: %s, Train: %.3f, Test: %.3f' % (k+1, train_score, test_score))
    
            title = "Learning Curve:\n thresh=" + str(threshold) + " c=" + str(c) + " co=" + str(n_components)
            plt.figure()
            plt.title(title)
            plt.xlabel("Iterations")
            plt.ylabel("Score")
    
            plt.grid()
    
            plt.plot(x_axis, train_scores, color="r",
                     label="Training score")
            plt.plot(x_axis, test_scores, color="g",
                     label="Testing score")
    
            plt.legend(loc="best")
            plt.savefig("lc_mlp_" + str(activation) + "_" + str(threshold) + "_" + str(c) + "_" + str(n_components) + ".png")
            #plt.show()

    print('done')



def nn4():
    train = pd.read_csv("train.csv")
    y = train.values[:,0]
    X = train.values[:,1:]

    # split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_train_std = X_train

    sc.fit(X_test)
    X_test_std = sc.transform(X_test)
    X_test_std = X_test
        
    clf = MLPClassifier(solver='sgd', activation='logistic', learning_rate='invscaling', max_iter=1000, verbose=True,
                        power_t=0.5, momentum=0.9, nesterovs_momentum=True, learning_rate_init=0.001)

    param_grid = {"power_t": np.linspace(0.1, 0.9, 9),
                  "momentum": np.linspace(0.0, 1.0, 11),
                  "nesterovs_momentum": [True, False],
                  "learning_rate_init": [1.0, 0.1, 0.01, 0.001, 0.0001]}

    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
    start = time()
    grid_search.fit(X_train_std, y_train)    

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_, 20)
    
    print('done')



def bug():
    X = np.random.rand(100,10)
    y = np.random.random_integers(0, 1, (100,))

    clf = MLPClassifier(max_iter=1, warm_start=True, verbose=True)
    for k in range(3):
        clf.fit(X, y)

if __name__ == '__main__':
    nn4()
