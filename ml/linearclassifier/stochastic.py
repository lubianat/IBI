from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import ml.linearclassifier.classifier as lc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
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
        

class StochasticLinearClassifier(lc.LinearClassifier):
    def __init__(self, eta, init_weights, should_stop,
         random_state=None, pvalue=1.0, nvalue=-1.0, total_error = list()):
        super().__init__(eta, init_weights, should_stop, pvalue, nvalue, total_error)
        
    @abstractmethod
    def compute_update(self, xi, yi):
        pass
    
    def fit(self, X, y):
                # set the initial weights
        self._w = lc._default_init_weights(self, len(X[0]))[1:][0]
        self._b = lc._default_init_weights(self, len(X[0]))[0]
        i=0
        self.total_error = list()
        #loop for a given number of epochs
        while i < self._should_stop:
            
            errors=0
            # go through each observation in the dataset AND update the weights for each obs
            for  xi,  target  in  zip(X,  y):
                update  = self.compute_update(xi, target)
                self._w += update  *  xi
                self._b += update
                errors += (target  != self.predict(xi))
            self.total_error.append(errors)
            #stop if the classification error in the training set is 0
            #if min(self.total_error) == 0 :
            #    break

            
            i+=1
            
        return self

    def partial_fit(self, X, y):
        # TODO: Implemente
        return self

class StochasticPerceptron(StochasticLinearClassifier):
    def compute_update(self, xi, target):
        update = self._eta * (target  - self.predict(xi))
        return update

class StochasticAdalineGD(StochasticLinearClassifier):
    def compute_update(self, xi, target):
        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = (target - output) 
        update = error*self._eta
        return update
    def activation(self, z):
        return z

class StochasticLogisticRegression(StochasticLinearClassifier):
    def compute_update(self, xi, target):
           net_input = self.net_input(xi)
           output = self.activation(net_input)
           error = (target - output) 
           update = error*self._eta
           return update

    def activation(self, z):
            return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
        
df1 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-1.data", sep=';', )
df2 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-2.data", sep=';', )

y = df1.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df1.iloc[:, [0,1]].values

hey = StochasticAdalineGD(eta = 0.0001, init_weights = 0, should_stop=3000)
hey.fit(X,y)
plot_decision_regions(X, y, classifier=hey)
plt.show()
plt.gcf().clear()