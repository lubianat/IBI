#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:07:16 2018

@author: lubianat
"""
import ml.linearclassifier.batch as batch
import ml.linearclassifier.stochastic as sls

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def plotError(classifier, X, y):
    #classifier - Classifier object (an instance of Stochastic Perceptron, for example) to use for testing the error
    #X - your 'n' samples and their 'm' variables
    #y - your 'n' classes, for your 'n' samples
    errors = 0
    for  xi,  target  in  zip(X,  y):
                errors += (target  != classifier.predict(xi))
    print('for this classifier the error is', errors  )    
    plot_decision_regions(X,y, classifier=classifier)
    plt.show()
    plt.gcf().clear()


#ALERT: the number of epochs and the eta is arbitrary! No grid search / hiperparameter optimization will be done due to long run times
#this resulted in logistic regression not reaching convergence! 
#convergence was not implemented in the final versions because the OVA classifier uses net inputs
#which are different in scale if differente number of epochs are used for adaline and logistic regression
def testAll(X,y, testX, testY):
    ok = ['Perceptron', 'AdalineGD', 'LogisticRegression']
    for bla, i in enumerate(ok):
        for ble, j in enumerate(['Batch','Stochastic']):
            if j == 'Batch':
                print(j+i)
                classifier = getattr(batch, j + i)
                instance = classifier(eta = 0.0001, init_weights = 0, should_stop=1000)
                instance.fit(X,y)
            if j == 'Stochastic':
               print(j+i)
               classifier = getattr(sls, j + i)
               instance = classifier(eta = 0.0001, init_weights = 0, should_stop=1000)
               instance.fit(X,y)
                         
            plotError(instance,testX,testY)
            plt.show()
            plt.gcf().clear()

train = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-4.data", sep=';', )
test = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/test-dataset-4.data", sep=';', )

y = train.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = train.iloc[:, [0,1]].values

testY=test.iloc[:, 2]
testY = np.where(testY == 'A', -1, 1)
testX = test.iloc[:, [0,1]].values
    
testAll(X,y,X,y)
testAll(X,y,testX,testY)


df42 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-4-ii.data", sep=';', )
test2 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/test-dataset-4-ii.data", sep=';', )

y = df42.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df42.iloc[:, [0,1]].values
   

testAll(X,y)
