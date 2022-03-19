from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib
from itertools import cycle
import pylab as pl


# This function has the goal of plotting
def plot_2d(data, target, target_names, x, y):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, x], data[target == i, y], c=c, label=label)


# This function is to count the number of instances for each class as well as to print the 32nd instance
def pandas_visualization(dataset):

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


"""
After Visualizing the dataset and trying different compositions of attributes the first and third attributes
The line that can separate than the data set into two sub sets clearly different can be the one passing by X(0, 4) and
Y(8, 1)
This function is to plot that the dataset the results discussed her
For the sack of maths this line has 'a' and 'b' as it's slope and intercept
b = 4
a = (1-4) / 8 = -3/8
"""


def plot_subclasses_2d(data, target, target_names):
    plot_2d(data, target, target_names, 0, 2)
    x, y = [0, 8], [4, 1]
    pl.plot(x, y)



if __name__ == '__main__':
    # Importing the iris dataset
    iris_dataset = datasets.load_iris()

    # It is an object with two attributes: data and target and others

    input = iris_dataset.data
    output = iris_dataset.target
    output_names = iris_dataset.feature_names

    # print(iris_dataset.data)
    # print(iris_dataset.target)

    # plot_2d(input, output, output_names, 0, 2)
    plot_subclasses_2d(input, output, output_names)
    pl.legend()
    pl.show()

    # pandas_visualization(iris_dataset)

