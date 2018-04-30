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
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    pass
