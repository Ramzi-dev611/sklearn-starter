"""
This script tries redo the same thing from the performance evaluation 2
but this time it will use a function from the sklearn library
"""

from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    data = iris_dataset.data
    target = iris_dataset.target
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
    nb.fit(x_train, y_train)
    accuracy_third = nb.score(x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1)
    nb.fit(x_train, y_train)
    accuracy_tenth = nb.score(x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    nb.fit(x_train, y_train)
    accuracy_fifth = nb.score(x_test, y_test)

    print('The error for the model trained on 66% of the data is ', 1 - accuracy_third)
    print('The error for the model trained on 90% of the data is ', 1 - accuracy_tenth)
    print('The error for the model trained on 80% of the data is ', 1 - accuracy_fifth)
