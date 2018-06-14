#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:50:42 2018

@author: lubianat
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-1.data", sep=';', )
df2 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-2.data", sep=';', )

plt.subplot(1, 2, 1)

y = df1.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df1.iloc[:, [0,1]].values
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.subplot(1, 2, 2)

y = df2.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df2.iloc[:, [0,1]].values
plt.scatter(X[:, 0], X[:, 1], c=y)

# Thus, dataset 1 is visually linearly separable

