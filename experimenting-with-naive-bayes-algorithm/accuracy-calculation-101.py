"""
This File has the script that:
    * generates a model of classification of the same dataset : iris
    * In this script the model is generated using the whole dataset
    * The model used is again Naive-Bayes
    * calculates the accuracy / error ratio of the model
"""
import numpy as np
import pandas as pd
from sklearn import datasets, naive_bayes


def iris_naive_bayes_accuracy_simple(dataset, model):
    acc = model.score(dataset.data, dataset.target)
    return acc


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


def iris_naive_bayes_get_wrong_predictions(dataset, clf):
    data = dataset.data[:]
    y = dataset.target
    p = clf.predict(data)
    dataframe = pd.DataFrame(np.c_[data, y, p], columns=[*dataset['feature_names'], 'y', 'p'])
    return dataframe[dataframe['y'] != dataframe['p']]


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    nb.fit(iris_dataset.data[:], iris_dataset.target[:])
    # accuracy = iris_naive_bayes_accuracy_simple(iris_dataset, nb)
    # accuracy = iris_naive_bayes_accuracy_long_code(iris_dataset, nb)
    accuracy = iris_naive_bayes_accuracy_short_code(iris_dataset, nb)
    miss_predicted = iris_naive_bayes_get_wrong_predictions(iris_dataset, nb)
    print('List of miss predicted instances: ')
    print(miss_predicted)
    print("Accuracy = ", accuracy)
    print("Error ration = ", 1 - accuracy)

