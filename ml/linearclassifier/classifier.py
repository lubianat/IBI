from abc import ABC, abstractmethod
from ml.core import Classifier 
import numpy as np

def _default_init_weights(classifier, N):
    np.random.seed(33)
    return (0, np.random.normal (loc=0, scale=0.01, size=N))

def _default_should_stop(classifier, i):
    return i < 10

class LinearClassifier(Classifier):
    def __init__(self, eta, init_weights, should_stop, pvalue=1.0, nvalue=-1.0, total_error=list()):
        self._eta = eta
        self._w = None
        self._b = 0
        self._init_weights = init_weights
        self._should_stop = should_stop
        self._pvalue = pvalue
        self._nvalue = nvalue
        self._epoch = 0
        self.total_error = total_error
    def net_input(self, x):
        return np.dot(x, self._w) + self._b

    def predict(self, x):
  
        return np.where(self.net_input(x) >= 0.0, self._pvalue, self._nvalue)

    @abstractmethod
    def fit(self, X, y):
        pass

    def bias(self):
        return self._b

    def nepochs(self):
        return self._epoch

    def pvalue(self):
        return self._pvalue

    def nvalue(self):
        return self._nvalue
