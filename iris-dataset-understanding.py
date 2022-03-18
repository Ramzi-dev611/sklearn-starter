from sklearn import datasets
import numpy as np
import pandas as pd


def pandas_visualization(dataset):
    # print(dataset.values())
    print("the dataset contains ", len(dataset.data), " instance")

    # Generating a pandas dataframe from the dict containing the data
    data_frame = pd.DataFrame(
        data=np.c_[dataset.data, dataset.target],
        columns=dataset['feature_names'] + ['target']
    )
    # printing for each class the number of records it contains in a dumb way
    for i in range(len(dataset['target_names'])):
        print("class ", i, "contains ", len(data_frame[data_frame.target == i]))


if __name__ == '__main__':
    # Importing the iris dataset
    iris_dataset = datasets.load_iris()

    # It is an object with two attributes: data and target and others
    print(iris_dataset.data)
    print(iris_dataset.target)

    # pandas_visualization(iris_dataset)
