from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib
from itertools import cycle
import pylab as pl


def plot_2d(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 1], data[target == i, 2], c=c, label=label)
    pl.legend()
    pl.show()


def pandas_visualization(dataset):
    # This function is to count the number of instances for each class as well as to print the 32nd instance

    print(dataset.values())
    print("the dataset contains ", len(dataset.data), " instance")

    # Generating a pandas dataframe from the dict containing the data
    data_frame = pd.DataFrame(
        data=np.c_[dataset.data, dataset.target],
        columns=dataset['feature_names'] + ['target']
    )
    # printing for each class the number of records it contains in a dumb way
    for i in range(len(dataset['target_names'])):
        print("class ", i, "contains ", len(data_frame[data_frame.target == i]))

    print(data_frame.iloc[31])


if __name__ == '__main__':
    # Importing the iris dataset
    iris_dataset = datasets.load_iris()

    # It is an object with two attributes: data and target and others

    data = iris_dataset.data
    target = iris_dataset.target
    target_names = iris_dataset.feature_names

    # print(iris_dataset.data)
    # print(iris_dataset.target)

    plot_2d(data, target, target_names)

    # pandas_visualization(iris_dataset)

