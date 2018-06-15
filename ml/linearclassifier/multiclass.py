import ml.linearclassifier.stochastic as slc
from ml.core import Classifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy as cp
import numpy as np
import pandas as pd



# classes and so


class MultiClassLinearClassifier(Classifier):
    # THIS MIGHT TAKE A WHILE FOR LARGE DATASETS! For the example, it takes a bit to have a stisfactory result. and still is not perfect
    def __init__(self, classifier_model=slc.StochasticPerceptron(eta = 0.001, init_weights = 0, should_stop=1000)):
        self._cm = classifier_model

    def fit(self, X, y):
        self.ova_classifiers = pd.DataFrame()
        for group in set(y):
            j = np.where(y == group, 1, -1)
            classifier = self._cm
            classifier.fit(X,j)
            weights = []
            weights.append(classifier._b)
            weights.append(classifier._w)
            self.ova_classifiers[group] = weights

    def predict(self, x):
        #return the best class. For each sample, check wwhich net input is the best
        best_classifier = []
        for samples in x:
            best = 0
            for c in self.ova_classifiers.columns:
                self._cm._b = self.ova_classifiers[c][0]
                self._cm._w = self.ova_classifiers[c][1:][1]
                local_input = self._cm.net_input(samples)
                if local_input > best:
                    best = local_input
                    classe = c
            best_classifier.append(classe)
        #returns a list with the best labels for each class
        return np.array(best_classifier)

df3 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-3.data", sep=';', )

mcc = MultiClassLinearClassifier()
y = df3.iloc[:, 2]
X = df3.iloc[:, [0,1]].values
mcc.fit(X,y)
mcc_p = mcc.predict(X)
mcc_p == np.array(y)

# auxiliary function
#the resolution may make it run for a really long time
def plot_decision_regions(X, y, classifier, resolution=0.05):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    for i, o in enumerate(np.unique(Z)):
         Z[Z==o] = i
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plot_decision_regions(X, y, classifier=mcc)
plt.show()
plt.gcf().clear()
