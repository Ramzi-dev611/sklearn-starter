"""
After the gaining knowledge on how to use some of the provided helper function
from the sklearn library, it is time to try and evaluate the performance of another algorithm:
Decision tree implementation of the same library
The dataset we are going to experiment on is still being the iris dataset (for the simplicity)
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from statistics import mean


def train_full_data(x, y, clf):
    clf.fit(x, y)
    return clf.score(x, y)


def split_test_train_stable(x, y, clf):
    accuracies = []
    for iteration in range(1000):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        clf.fit(x_train, y_train)
        accuracies.append(clf.score(x_test, y_test))
    return mean(accuracies)


# for the sake of just running a simple example here will fix the number of folds on 10
def cross_validation_train(x, y, clf):
    return mean(cross_val_score(clf, x, y, cv=10))


if __name__ == '__main__':
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target
    decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    accuracy_full = train_full_data(data, target, decision_tree)
    accuracy_split = split_test_train_stable(data, target, decision_tree)
    accuracy_cv = cross_validation_train(data, target, decision_tree)
    plt.scatter(
        ['full data training', 'split train and test', 'cross validation'],
        [accuracy_full, accuracy_split, accuracy_cv]
    )
    plt.show()
