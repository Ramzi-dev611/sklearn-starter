"""
This File has the script that:
    * generates a model of classification of the same dataset : iris
    * In this script the model is generated using the whole dataset
    * The model used is again Naive-Bayes
    * calculates the accuracy / error ratio of the model
"""
from sklearn import datasets, naive_bayes
import numpy as np


def iris_naive_bayes_accuracy_simple(dataset, model):
    accuracy = model.score(dataset.data, dataset.target)
    return accuracy


def iris_naive_bayes_accuracy_long_code(dataset, model):
    y = dataset.target
    p = model.predict(dataset.data[:])
    e = 0
    for index, _ in enumerate(p):
        if p[index] != y[index]:
            e += 1
    return 1 - (e/len(p))


def iris_naive_bayes_accuracy_short_code(dataset, model):
    y = dataset.target
    p = model.predict(dataset.data[:])
    return len((p-y)[p-y == 0])/len(y)


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    nb.fit(iris_dataset.data[:], iris_dataset.target[:])
    # accuracy = iris_naive_bayes_accuracy_simple(iris_dataset, nb)
    # accuracy = iris_naive_bayes_accuracy_long_code(iris_dataset, nb)
    accuracy = iris_naive_bayes_accuracy_short_code(iris_dataset, nb)
    print("Accuracy = ", accuracy)
    print("Error ration = ", 1 - accuracy)
