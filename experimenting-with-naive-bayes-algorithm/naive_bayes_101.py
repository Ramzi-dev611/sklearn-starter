from sklearn import naive_bayes, datasets
import numpy as np


def iris_naive_bayes_full_data(dataset):
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    nb.fit(dataset.data[:-1], dataset.target[:-1])
    # taking the only one element which is the 31st instance to predict
    p31 = nb.predict(dataset.data[31:32])
    print(p31)
    # building a numpy 2d array containing only the last instance
    arr = np.reshape(dataset.data[-1], (1, 4))
    plast = nb.predict(arr)
    print(plast)
    prediction = nb.predict(dataset.data[:])
    print(prediction)


def iris_naive_bayes_partial_data(dataset):
    nb = naive_bayes.MultinomialNB(fit_prior=True)
    nb.fit(dataset.data[:99], dataset.target[:99])
    prediction = nb.predict(dataset.data[100: -1])
    print(prediction)
    pred_arr = np.array(prediction)
    actual_arr = np.array(dataset.target[100:-1])
    acc = 100*(len(pred_arr[pred_arr == actual_arr])/51)
    print("Accuracy = ", acc)
    # Terrible results since the dataset splitting here didn't include any instance classified as the second class


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    iris_naive_bayes_full_data(iris_dataset)
    iris_naive_bayes_partial_data(iris_dataset)
