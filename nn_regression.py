import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt
import numpy
from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    y_pred = nn.predict(x)
    mse = mean_squared_error(y, y_pred)
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    n_hidden = 8
    trained_regressor = MLPRegressor(hidden_layer_sizes=(n_hidden, ), activation='logistic', solver='lbfgs', alpha=0, max_iter=200, random_state=1)
    trained_regressor = trained_regressor.fit(x_train,y_train)
    y_pred_train = trained_regressor.predict(x_train)
    y_pred_test = trained_regressor.predict(x_test)
    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    n_iterations = 10
    train_mses = numpy.zeros(n_iterations)
    test_mses = numpy.zeros(n_iterations)
    for i in range(n_iterations):
        trained_regressor = MLPRegressor(warm_start = False, hidden_layer_sizes=(8, ), activation='logistic', solver='lbfgs', alpha=0, max_iter=200, random_state=random.randint(1,100))
        trained_regressor = trained_regressor.fit(x_train,y_train)
        train_mses[i] = calculate_mse(trained_regressor,x_train,y_train)
        test_mses[i] = calculate_mse(trained_regressor,x_test,y_test)
    
    print ("Train_mses")
    print (train_mses)
    print ("Train_min")
    print (train_mses.min())
    print ("Train_max")
    print (train_mses.max())
    print ("Train_std")
    print (train_mses.std())

    print ("Test_mses")
    print (test_mses)
    print ("Test_min")
    print (test_mses.min())
    print ("Test_max")
    print (test_mses.max())
    print ("Test_std")
    print (test_mses.std())

    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    Use max_iter = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_hidden_neurons_list = [1,2,3,4,6,8,12,20,40]
    n_iterations = 10
    train_mses = numpy.zeros((9,n_iterations))
    test_mses = numpy.zeros((9,n_iterations))
    r = 0
    for n_hidden_neuron in n_hidden_neurons_list :
        for i in range(n_iterations):
            trained_regressor = MLPRegressor(hidden_layer_sizes=(n_hidden_neuron, ), activation='logistic', solver='lbfgs', alpha=0,tol= 1e-8, max_iter=10000,random_state=i)
            trained_regressor = trained_regressor.fit(x_train,y_train)
            train_mses[r][i] = calculate_mse(trained_regressor,x_train,y_train)
            test_mses[r][i] = calculate_mse(trained_regressor,x_test,y_test)
        r = r + 1
    plot_mse_vs_neurons(train_mses, test_mses, n_hidden_neurons_list)

    # trained_regressor = MLPRegressor(hidden_layer_sizes=(40, ), activation='logistic', solver='lbfgs', alpha=0,tol= 1e-8, max_iter=10000,random_state=1)
    # trained_regressor = trained_regressor.fit(x_train,y_train)
    # y_pred_train = trained_regressor.predict(x_train)
    # y_pred_test = trained_regressor.predict(x_test)
    # plot_learned_function(40, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    Use n_iterations = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    hidden_neurons_list = [2,8,40]
    n_iterations = 10000

    train_mses = numpy.zeros((3,n_iterations))
    test_mses = numpy.zeros((3,n_iterations))
    r = 0
    for n_hidden_neuron in hidden_neurons_list :
        for i in range(n_iterations):
            trained_regressor = MLPRegressor(warm_start = True, hidden_layer_sizes=(n_hidden_neuron, ), activation='logistic', solver='lbfgs', alpha=0,tol= 1e-8, max_iter=1,random_state=i)
            trained_regressor = trained_regressor.fit(x_train,y_train)
            train_mses[r][i] = calculate_mse(trained_regressor,x_train,y_train)
            test_mses[r][i] = calculate_mse(trained_regressor,x_test,y_test)
        r = r + 1
    plot_mse_vs_neurons(train_mses, test_mses, hidden_neurons_list)

    ## TODO
    pass


def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    # alph = [pow(10,−8),pow(10,−7),pow(10,−6),pow(10,−5),pow(10,−4),pow(10,−3),pow(10,−2),pow(10,−1),1,10,100]
    # n_iterations = 10000

    # train_mses = numpy.zeros((3,n_iterations))
    # test_mses = numpy.zeros((3,n_iterations))
    # r = 0
    # for n_hidden_neuron in hidden_neurons_list :
    #     for i in range(n_iterations):
    #         trained_regressor = MLPRegressor(warm_start = True, hidden_layer_sizes=(n_hidden_neuron, ), activation='logistic', solver='lbfgs', alpha=0,tol= 1e-8, max_iter=1,random_state=i)
    #         trained_regressor = trained_regressor.fit(x_train,y_train)
    #         train_mses[r][i] = calculate_mse(trained_regressor,x_train,y_train)
    #         test_mses[r][i] = calculate_mse(trained_regressor,x_test,y_test)
    #     r = r + 1
    # plot_mse_vs_neurons(train_mses, test_mses, hidden_neurons_list)
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass


def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    ## TODO
    pass
