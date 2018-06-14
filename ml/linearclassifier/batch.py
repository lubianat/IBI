from abc import ABC, abstractmethod
import numpy as np
import ml.linearclassifier.classifier as lc


# eta is the learning rate (float)
# should_stop is the number of epochs the algorithm has to run before it should stop
# init_weights can be like anything, the object sets it anyways

class BatchLinearClassifier(lc.LinearClassifier):
    def __init__(self, eta,    init_weights, should_stop, pvalue=1.0, nvalue=-1.0,total_error= list()):
        super().__init__(eta, init_weights, should_stop, pvalue, nvalue,total_error)
        

    @abstractmethod
    def compute_update(self, X, y):
        pass

    def fit(self, X, y):
        # set the initial weights
        self._w = lc._default_init_weights(self, len(X[0]))[1:][0]
        self._b = lc._default_init_weights(self, len(X[0]))[0]

        i=0
        self.total_error = list()
        #loop for a given number of epochs
        while i < self._should_stop:
            deltaW  = 0
            deltaB = 0
            errors = 0
                        # go through each observation in the dataset
            for  xi,  target  in  zip(X,  y):
                update  = self.compute_update(xi, target)
                deltaW += update  *  xi
                deltaB += update
                errors += (target  != self.predict(xi))
            self.total_error.append(errors) 
            # reset the weights and bias fo next epoch
            
            self._w = self._w + deltaW
            self._b = self._b + deltaB
            i+=1
        
        return self


class BatchPerceptron(BatchLinearClassifier):
    def compute_update(self, xi, target):
        update = self._eta * (target  - self.predict(xi))
        return update


class BatchAdalineGD(BatchLinearClassifier):

    def compute_update(self, xi, target):
        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = (target - output)
        update = error*self._eta
        return update

    def activation(self, z):
            return z

class BatchLogisticRegression(BatchLinearClassifier):
    def compute_update(self, xi, target):
           net_input = self.net_input(xi)
           output = self.activation(net_input)
           error = (target - output)
           update = error*self._eta
           return update

    def activation(self, z):
            return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
