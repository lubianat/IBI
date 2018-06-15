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

df1 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-1.data", sep=';', )
df2 = pd.read_csv("/home/lubianat/Documentos/Disciplines/IBI5---/tarefa_01/dataset/dataset-2.data", sep=';', )

y = df1.iloc[:, 2]
y = np.where(y == 'A', -1, 1)
X = df1.iloc[:, [0,1]].values
plt.scatter(X[:, 0], X[:, 1], c=y)

net = sls.StochasticPerceptron(eta = 0.0001, init_weights = 0, should_stop=300)
net.fit(X,y)

# get decisions regions
plt.subplot(1, 2, 1)
plot_decision_regions(X,y, classifier=net)

# plot errors to see convergence
plt.subplot(1, 2, 2)
plt.scatter(np.asarray(list(range(1,len(net.total_error)+1))),net.total_error)


# get convergence step
def convergencePoint(total_error):
    convergence_point=0
    for index, element in enumerate(total_error):
        if element < total_error[index-1]:
            convergence_point = index+1
    return convergence_point

print('the convergence step  for the Stochastic Perceptron is:', convergencePoint(net.total_error))
plt.show()
plt.gcf().clear()
