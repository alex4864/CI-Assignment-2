from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
import numpy as np

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    # parse target2 2nd column
    pose2 = []
    for target in target2:
        pose2.append(target[1])

    mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=6)
    print("===========fit started===========")
    mlp.fit(input2, pose2)
    print("===========fit finished===========")
    print("classes_: ", mlp.classes_)
    print("n_layers_: ", mlp.n_layers_)
    plot_hidden_layer_weights(mlp.coefs_[0])

    print("===========predict started===========")
    prediction = mlp.predict(input2)
    print("===========predict finished===========")
    cnf_matrix = confusion_matrix(pose2, prediction)
    print(cnf_matrix)
    return


def ex_2_2(input1, target1, input2, target2):
    individualTarget1 = target1[:, 0]
    individualTarget2 = target2[:, 0]

    classifiers = []
    trainingScores = []
    testScores = []
    for i in range(10):
        classifiers.append(MLPClassifier(activation='tanh', hidden_layer_sizes=20, max_iter=1000, random_state=i))

    for mlp in classifiers:
        mlp.fit(input1, individualTarget1)
        trainingScores.append(mlp.score(input1, individualTarget1))
        testScores.append(mlp.score(input2, individualTarget2))

    plot_histogram_of_acc(trainingScores, testScores)