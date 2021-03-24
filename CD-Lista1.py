#QUESTÃO 4 - LETRA A

import numpy as np

def mediana(x):
    x.sort()
    n = len(x)
    if n % 2 == 1:
        return x[n // 2]
    else:
        return (x[n // 2 - 1] + x[n // 2]) / 2

V1 = [1, 15, 20, 578, 799]
print("Media: ",np.mean(s),"\nMediana: ", mediana(s))

#QUESTÃO 4 - LETRA B

import numpy as np

def mediana(x):
    x.sort()
    n = len(x)
    if n % 2 == 1:
        return x[n // 2]
    else:
        return (x[n // 2 - 1] + x[n // 2]) / 2

V2 = [10, 9, 17, 16, 15]
print("Media: ",np.mean(V2),"\nMediana: ", mediana(V2))

#QUESTÃO 5 - LETRA A

import seaborn as sns
V1 = [1, 15, 20, 578, 799]

sns.displot(V1)

V2 = [10, 9, 17, 16, 15]

sns.displot(V2)

#QUESTÃO 5 - LETRA B

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

V1 = [1, 15, 20, 578, 799]

mediaV1 = np.mean(V1)
sigma = 0.1

s = np.random.normal(mediaV1, sigma, 1000)
sns.histplot(s)
V2 = [10, 9, 17, 16, 15]

mediaV2 = np.mean(V2)
sigma = 0.1

s = np.random.normal(mediaV2, sigma, 1000)
sns.histplot(s)



#QUESTÃO 10

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

nomes = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
df = pd.read_csv('iris.data', names = nomes)

X = df[df.columns.difference(['Class'])].values
y = df['Class'].values

iris_classificador = KNeighborsClassifier(n_neighbors = 3)
iris_classificador2 = KNeighborsClassifier(n_neighbors = 5)
iris_classificador3 = KNeighborsClassifier(n_neighbors = 10)
iris_classificador.fit(X, y)
iris_classificador2.fit(X, y)
iris_classificador3.fit(X, y)

scores_dt = cross_val_score(iris_classificador, X, y, scoring='accuracy', cv=5)
print("Utilizando o K-fold cross-validation:")
print(scores_dt.mean())

y_true = y
y_pred = iris_classificador.predict(X)
y_pred2 = iris_classificador2.predict(X)
y_pred3 = iris_classificador3.predict(X)
print("Utilizando o F-measure com 3:")
print(f1_score(y_true, y_pred, average='macro'))
print("Utilizando o F-measure com 5:")
print(f1_score(y_true, y_pred2, average='macro'))
print("Utilizando o F-measure com 10:")
print(f1_score(y_true, y_pred3, average='macro'))