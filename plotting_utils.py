"""
A util module containing methods that help in plotting data for basic_sarsamax module.

Created by Patryk Stradomski, s16716
"""
import matplotlib.pyplot as plt
import numpy as np


def epsilon_plot(epsilon, epsilon_decay, minimal_epsilon, number_of_iterations):
    """
    This method generates the values of decaying epsilon by iteration
    :param epsilon: Initial value of epsilon parameter
    :param epsilon_decay: The value by which the epsilon is multiplied each episode
    :param minimal_epsilon: Minimal value of epsilon, Epsilon won't go lower than that value during decay
    :param number_of_iterations: number of iterations of the decay
    :return: list of values that represent value of epsilon during a given step - (0, x)
    """
    values = []
    for _ in range(number_of_iterations):
        values.append(epsilon)
        epsilon = max(epsilon * epsilon_decay, minimal_epsilon)
    return values


def plot_scores_with_epsilon_decay(scores, initial_epsilon, epsilon_decay, minimal_epsilon):
    """
    A util method that plots scores over time and draws an line representing the decay of epsilon over time.
    If used in pair with scores and epsilon values of a sarsamax training
    the score should stabilise as epsilon approaches 0.
    :param scores: list of scores to be plotted
    :param initial_epsilon: Initial value of epsilon parameter
    :param epsilon_decay: The value by which the epsilon is multiplied each episode
    :param minimal_epsilon: Minimal value of epsilon, Epsilon won't go lower than that value during decay
    """
    plt.plot(scores, label='Episode Score')
    plt.plot(epsilon_plot(initial_epsilon, epsilon_decay, minimal_epsilon, len(scores)), '-r', label='Epsilon value')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right", frameon=True)
    plt.show()


def plot_scores_with_regression_line(scores):
    """
    A util method that plots scores over time and draws a regression line according to the scores behavior
    :param scores: list of scores to be plotted
    """
    plt.style.use('seaborn-whitegrid')
    episodes = np.arange(start=0, stop=len(scores))
    regression_line = np.poly1d(np.polyfit(episodes, scores, 5))
    plt.plot(scores, label='Episode Score')
    plt.plot(episodes, regression_line(episodes), label='Regression line')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right", frameon=True)
    plt.show()
