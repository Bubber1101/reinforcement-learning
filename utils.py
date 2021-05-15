import matplotlib.pyplot as plt
import numpy as np


def epsilon_plot(beggining, decay, min, x):
    values = np.zeros(x)
    epsilon = beggining

    for i in range(x):
        epsilon = max(epsilon * decay, min)
        values[i] = epsilon
    return values


def plot_scores_with_epsilon_decay(scores, number_of_episodes):
    plt.plot(scores, label='Episode Score')
    plt.plot(epsilon_plot(1, 0.75, 0.01, 100), '-r', label='Epsilon value')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right")


def plot_scores_with_regression_line(scores: np.ndarray):
    """
    A util method that plots scores over time and draws
    :param scores: list of scores to be plotted
    :return:
    """
    plt.style.use('seaborn-whitegrid')
    x = np.arange(start=0, stop=len(scores))
    regression_line = np.poly1d(np.polyfit(x, scores, 5))
    plt.plot(scores, label='Episode Score')
    plt.plot(x, regression_line(x), label='Regression line')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right")
    plt.show()
