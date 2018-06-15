import ml.linearclassifier.batch as batch
import ml.linearclassifier.stochastic as sls
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ml.mlplot as mlplot
from ex13 import plotError
from ex13 import testAll

df42 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-4-ii.data", sep=';', )
test2 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/test-dataset-4-ii.data", sep=';', )

y = df42.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df42.iloc[:, [0,1]].values

testY2 =test2.iloc[:, 2]
testY2 = np.where(testY2 == 'A', -1, 1)
testX2 = test2.iloc[:, [0,1]].values

print('for the training set with have the following results:\n')
testAll(X,y,X,y, 'training - df4 II')

print('for the test set with have the following results:\n')
testAll(X,y,testX2,testY2, 'test - df4 II')
