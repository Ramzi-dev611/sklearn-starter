"""
As we did for the first performance evaluation
in this script we will try to split the dataset into 2 datasets
one is going to be for training the model
the other one is going to be for testing
the model generated is naive bayes again
and we we will keep on working with the iris dataset
"""
import random

from sklearn import datasets, naive_bayes
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


def split(s, training_proportion):
    data = s.data
    target = s.target
    n = len(data)
    indexes = np.random.choice(range(len(data)), len(data))
    data_s1 = [data[index] for index in indexes[:int(training_proportion * n)]]
    data_s2 = [data[index] for index in indexes[int(training_proportion * n):-1]]
    target_s1 = [target[index] for index in indexes[:int(training_proportion * n)]]
    target_s2 = [target[index] for index in indexes[int(training_proportion * n):-1]]
    return data_s1, target_s1, data_s2, target_s2


def test(s, clf, training_proportion, number_tests, plot_option=False):
    iteration_error = []
    for iteration in range(number_tests):
        train_data, train_target, test_data, test_target = split(s, training_proportion)
        clf.fit(train_data, train_target)
        accuracy = clf.score(test_data, test_target)
        iteration_error.append(1 - accuracy)
    if plot_option:
        plt.scatter(range(number_tests), iteration_error)
        plt.show()
    return mean(iteration_error)


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    # test_number = random.choice([10, 50, 100, 200, 500, 1000])
    test_number = 10
    data_portion_third = 2/3
    data_portion_tenth = 9/10
    total_error_third = mean(test(iris_dataset, nb, data_portion_third, test_number) for iteration in range(20))
    total_error_tenth = mean(test(iris_dataset, nb, data_portion_tenth, test_number) for iteration in range(20))
    print('For a training data presenting ', data_portion_third*100, '% of the dataset')
    print('Error = ', total_error_third)
    print('For a training data presenting ', data_portion_tenth*100, '% of the dataset')
    print('Error = ', total_error_tenth)
    # plot example
    test(iris_dataset, nb, data_portion_third, test_number, plot_option=True)


"""
For the Question b: since the dataset splitting is random, each time the execution gets us a different 
error estimated sometimes better than for the case of having all the dataset used for training sometimes not
"""

"""
When having the test made n times and having the process of testing n times 20 times
we get an average error ration of proximately 0.238
"""

"""
The comparison between the 1/3 for test model and the 1/10 for test will be the one left based on the number of test = 10
"""

