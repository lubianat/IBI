from abc import ABC, abstractmethod
import numpy as np
import ml.linearclassifier.classifier as lc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
        # check if the weights were already initialized, if not, run fit
        # if they were, run one round of update with the online wieghts and target classes
        if not isinstance(self._w, np.ndarray) :
            self._w = lc._default_init_weights(self, len(X[0]))[1:][0]
            self._b = lc._default_init_weights(self, len(X[0]))[0]
        else:
        #check if input has one ore more observations and then update
            if isinstance(y, np.ndarray) :
                for xi, target in zip(X, y):                 
                    update  = self.compute_update(xi, target)
                    self._w += update  *  xi
                    self._b += update
        #redundancy for  the sake of clarity
            if isinstance(y, np.int64):
                update  = self.compute_update(X, y)
                self._w += update  *  X
                self._b += update
            
        
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
           cost = (target*(np.log(output)) + ((1 - target)*(np.log(1 - output))))
           return update

    def activation(self, z):
        # 'clips' if z has a really high absolute value
            return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
            return np.where(self.activation(self.net_input(X))>= 0.5, 1, 0)

