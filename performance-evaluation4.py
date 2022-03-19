"""
    Dataset : Iris
    Algorithm: Naive Bayes
    Testing technic: Cross Validation
    Goal writing a script that trains a model on the iris dataset and getting out with least error estimation possible
"""

from sklearn import datasets, naive_bayes
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev
import matplotlib.pyplot as plt


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    data = iris_dataset.data
    target = iris_dataset.target
    naive = naive_bayes.MultinomialNB(fit_prior=True)
    iterations = range(3, 11)
    scores = [cross_val_score(naive, data, target, cv=iteration) for iteration in iterations]
    mean_errors = [mean(score) for score in scores]
    std_dev_errors = [stdev(score) for score in scores]
    print('The mean error of cross validation trained model using 3 folds is %0.3f' % (mean_errors[0]))
    print('The mean error of cross validation trained model using 5 folds is %0.3f' % (mean_errors[2]))
    print('The mean error of cross validation trained model using 8 folds is %0.3f' % (mean_errors[5]))
    plt.scatter(iterations, mean_errors)
    plt.scatter(iterations, std_dev_errors)
    plt.legend(['Mean Errors', 'Standard deviation of Errors'])
    plt.show()