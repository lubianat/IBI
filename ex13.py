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
import ml.mlplot as mlplot

def plotError(classifier, X, y):
    #classifier - Classifier object (an instance of Stochastic Perceptron, for example) to use for testing the error
    #X - your 'n' samples and their 'm' variables
    #y - your 'n' classes, for your 'n' samples
    errors = 0
    for  xi,  target  in  zip(X,  y):
                errors += (target  != classifier.predict(xi))
    print('for this classifier the number of missclassified samples is', errors  )
    mlplot.plot_decision_regions(X,y, classifier=classifier)
    plt.show()
    plt.gcf().clear()


#ALERT: the number of epochs and the eta are arbitrary! No grid search / hiperparameter optimization was done, due to long computational times

def testAll(X,y, testX, testY, name):
    ok = ['Perceptron', 'AdalineGD', 'LogisticRegression']
    for bla, i in enumerate(ok):
        if i == 'LogisticRegression':
            y = np.where(y == -1, 0, 1)
            testY = np.where(testY == -1, 0, 1)
        for ble, j in enumerate(['Batch','Stochastic']):
            if j == 'Batch':
                print(j+i+' '+name)
                classifier = getattr(batch, j + i)


            if j == 'Stochastic':
               print(j+i+' '+name)
               classifier = getattr(sls, j + i)

            nepochs = 100
            eta = 0.0001
            instance = classifier(eta = eta, init_weights = 0, should_stop=nepochs)
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

print('for the training set with have the following results:\n')
testAll(X,y,X,y, '-training df4')
print('for the test set with have the following results:\n')
testAll(X,y,testX,testY,'-test df4')
