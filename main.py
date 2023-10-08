import iris
import titanic
import diabetes
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from KNN import Classification_KNN
from KNN import Regression_KNN


"""
# Считывание данных (титаник)
X = titanic.read()
X, target = titanic.preparing(X)

# Разделение выборки (титаник)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = titanic.divide(X, target, train_len, validation_len, test_len)
"""

"""
# Считывание данных (ирисы)
X, target = load_iris(return_X_y=True)
x1, target1 = X[0], target[0]
x2, target2 = X[1], target[1]
X, target = X[2:], target[2:]

# Разделение выборки (ирисы)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = iris.splitting(X, target, train_len, validation_len, test_len)
"""

"""
accuracy = []
neighbours = []
# Валидация гиперпараметров
for i in range(3, 15):
    classifier = Classification_KNN(k=i)
    classifier.fit(Train, T_Train)
    prediction = classifier.prediction(Validation)
    acc = np.sum(prediction == T_Validation) / len(T_Validation)
    accuracy.append(acc)
    neighbours.append(i)
    print(acc)

print(f"Best accuracy on validation dataset: {max(accuracy)}")
print(f"Best quantity of neighbours: {neighbours[accuracy.index(max(accuracy))]}")

classifier = Classification_KNN(k=neighbours[accuracy.index(max(accuracy))])
classifier.fit(Train, T_Train)
prediction = classifier.prediction(Test)
print(f"Accuracy on test dataset: {np.sum(prediction == T_Test) / len(T_Test)}")

print()
"""

"""
# Вектора уверенности (ирисы)
print(f"Real class: {target1}")
print(classifier.one_prediction(x1))

print()

print(f"Real class: {target2}")
print(classifier.one_prediction(x2))
"""

# Задача регрессии
X, target = load_diabetes(return_X_y=True)

# Разделение выборки (диабет)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = diabetes.splitting(X, target, train_len, validation_len, test_len)

accuracy = []
neighbours = []
# Валидация гиперпараметров
for i in range(3, 15):
    classifier = Regression_KNN(k=i)
    classifier.fit(Train, T_Train)
    prediction = classifier.prediction(Validation)
    acc = 1 - np.sum((prediction - T_Validation)**2) / np.sum((T_Validation.mean() - T_Validation)**2)
    accuracy.append(acc)
    neighbours.append(i)
    print(acc)

print(f"Best accuracy on validation dataset: {max(accuracy)}")
print(f"Best quantity of neighbours: {neighbours[accuracy.index(max(accuracy))]}")

classifier = Regression_KNN(k=neighbours[accuracy.index(max(accuracy))])
classifier.fit(Train, T_Train)
prediction = classifier.prediction(Test)
print(f"Accuracy on test dataset: {1 - np.sum((prediction - T_Test)**2) / np.sum((T_Test.mean() - T_Test)**2)}")