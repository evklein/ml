import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.01, num_iterations = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        print(X.shape)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.1, size = X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.num_iterations):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(x_i))
                print('Predictions', self.predict(x_i), target, 'Update', update)
                self.w_ += update * x_i
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print(self.w_, self.b_, errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

import pandas as pd
iris = pd.read_csv('data/iris/iris.data', header = None, encoding = 'utf-8')

import matplotlib.pyplot as plt
y = iris.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = iris.iloc[0:100, 0:4].values

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 's', label = 'Versicolor')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.legend(loc = 'upper left')
plt.show()

perceptron = Perceptron(learning_rate = 0.01, num_iterations = 10)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.show()