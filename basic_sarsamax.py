"""
Performs a Q-Learning training on a Unity ML-Agents environment and plots the results.
The module was created with "Basic" environment in mind but will work with any other that shares it's interface.
See https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#basic
Calibration values can be passed to override the defaults.

Created by Patryk Stradomski, s16716
"""

import argparse
import os

import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException

from plotting_utils import plot_scores_with_regression_line, plot_scores_with_epsilon_decay


def basic_q_learning(
        filepath,
        discount_factor,
        learning_rate,
        epsilon,
        epsilon_decay,
        epsilon_min,
        number_of_episodes):
    """
    This performs a sarsamax training on "Basic" environment (or it's alteration) passed by filepath.
    The environment was prepared in unity and the script uses
    Unity ML-Agents Python Interface(https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs)
    in order to communicate with the executable.

    :param filepath: Path to an executable containing a "Basic" unity environment
    :param discount_factor: Sarsamax calibration parameter - discount factor of the function
    :param learning_rate: Sarsamax calibration parameter - learning rate of the function
    :param epsilon: The initial value of the variable checked against during explore v. exploit decision
    It should be between 0 and 1
    It is getting multiplied by epsilon_decay each episode
    :param epsilon_decay: The value by which the epsilon is multiplied each episode
    If no decay is necessary pass 1 here
    :param epsilon_min: Minimal value of epsilon, Epsilon won't go lower than that value during decay
    :param number_of_episodes: Number of episodes that the training should last for
    Episode ends when a reward is reached
    :return: scores - a list of scores accumulated in each episode,
     q_table - an array representing the agents Q-table
    """
    unity_environment = UnityEnvironment(filepath)
    environment = UnityToGymWrapper(unity_environment)
    possible_states = environment.observation_space.shape[0]
    possible_actions = environment.action_space.n
    q_table = np.ones((possible_states, possible_actions))
    scores = []

    for _ in range(number_of_episodes):
        score = 0
        state = environment.reset().argmax()  # Translation from list to integer
        done = False
        while not done:
            action = choose_explore_exploit(q_table, state, epsilon, environment.action_space)
            next_state, reward, done, _ = environment.step(action)
            if next_state is not None:
                next_state = next_state.argmax()  # Translation from list to integer
            score = score + reward
            q_table[state][action] = calculate_q_value(q_table,
                                                       reward,
                                                       learning_rate,
                                                       discount_factor,
                                                       q_table[state][action],
                                                       next_state)
            if not done:
                state = next_state

        scores.append(score)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return scores, q_table


def calculate_q_value(q_table,
                      reward,
                      learning_rate,
                      discount_factor,
                      current_q_value,
                      next_state):
    """
    Runs the sarsamax algorithm and returns new Q-Value for given state
    :param q_table: The Q-table from which the Q-value is being calculated
    :param reward:award received after last action
    :param learning_rate: float (0,1] Sarsamax calibration parameter - learning rate of the function
    :param discount_factor: float(0,1] Sarsamax calibration parameter - discount factor of the function
    :param current_q_value: Current Q-Value for considered state:action pair
    :param next_state: The state that came after the performed action
    :return:
    """
    expected_next_q_value = get_max_q_value_for_state(next_state, q_table)

    return current_q_value + learning_rate * (reward + discount_factor * expected_next_q_value - current_q_value)


def get_max_q_value_for_state(state, q_table):
    """
    :param state: State for which the actions should be checked
    :param q_table: Q_table with the state:action pairs
    :return: the biggest Q_value from the possible state:action pairs in given Q_table
    """
    if state is None:
        return 0
    else:
        return np.max(q_table[state])


def choose_explore_exploit(q_table, state, epsilon, action_space: UnityToGymWrapper.action_space):
    """
    :param q_table: Q_table with state:action pairs
    :param state: State for which the action should be selected
    :param epsilon: Float value belonging to (0,1)
    :param action_space: UnityToGymWrapper.action_space of the environment
    :return: An action to be performed, belonging to the action_space
    """
    rand = np.random.rand()
    if rand < epsilon:
        return action_space.sample()
    else:
        return get_action(q_table, state)


def get_action(q_table, state):
    """
    In SARSAMAX the action is chosen best on the highest q-value.
    This method finds the highest q-value for the given state and return the index of the action in the q-table.
    Since the q-table is created based on the env.action_space and env.observation_space
    the returned index will directly correspond to the appropriate action.

    :param q_table: The Q-table from which the action should be derived.
    :param state: Current observation of the environment for which the most suitable action should be found
    :return: an integer representing the action to be performed
    """
    return np.argmax(q_table[state])


def existing_file(filepath):
    """
    Checks if there is a file under the provided path
    """
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError("{0} is not a path to a file".format(filepath))
    return filepath


def epsilon_parameter(argument):
    """
    Checks if value is a float and belongs to [0,1]
    """
    try:
        argument = float(argument)
    except ValueError:
        raise argparse.ArgumentTypeError("{0} illegal argument passed, should be a float".format(argument))

    if argument < 0 or argument > 1:
        raise argparse.ArgumentTypeError("{0} illegal argument passed, should be between [0, 1]".format(argument))
    return argument


def sarsa_parameter(argument):
    """
    Checks if value is a float and belongs to (0,1]
    """
    try:
        argument = float(argument)
    except ValueError:
        raise argparse.ArgumentTypeError("{0} illegal argument passed, should be a float".format(argument))

    if argument <= 0 or argument > 1:
        raise argparse.ArgumentTypeError("{0} illegal argument passed, should be between (0, 1]".format(argument))
    return argument


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sarsamax implementation on 'Basic' unity environment")
    parser.add_argument("-f", "--file",
                        dest="filepath", required=True, type=existing_file,
                        help="A path to an executable file containing a version of 'Basic' environment")

    parser.add_argument("-d", "--discount_factor", default=0.99,
                        dest="discount_factor", required=False, type=sarsa_parameter,
                        help="Sarsamax calibration parameter - discount factor of the function")

    parser.add_argument("-l", "--learning_rate", default=0.99,
                        dest="learning_rate", required=False, type=sarsa_parameter,
                        help="Sarsamax calibration parameter - learning rate of the function")

    parser.add_argument("-e", "--epsilon", default=1,
                        dest="epsilon", required=False, type=epsilon_parameter,
                        help="The initial value of the variable checked against during explore v. exploit decision.")

    parser.add_argument("-ed", "--epsilon_decay", default=0.975,
                        dest="epsilon_decay", required=False, type=epsilon_parameter,
                        help="The value by which the epsilon is multiplied each episode.")

    parser.add_argument("-em", "--epsilon_min", default=0.01,
                        dest="epsilon_min", required=False, type=epsilon_parameter,
                        help="Minimal value of epsilon, Epsilon won't go lower than that value during decay.")

    parser.add_argument("-n", "--number_of_episodes", default=100,
                        dest="number_of_episodes", required=False, type=int,
                        help="Number of episodes that the training should last for.")
    args = parser.parse_args()

    try:
        score_summary, populated_q_table = basic_q_learning(args.filepath, args.discount_factor, args.learning_rate,
                                                            args.epsilon, args.epsilon_decay, args.epsilon_min,
                                                            args.number_of_episodes)

        plot_scores_with_regression_line(score_summary)
        plot_scores_with_epsilon_decay(score_summary, args.epsilon, args.epsilon_decay, args.epsilon_min)
        print(populated_q_table)
    except UnityEnvironmentException as error:
        print('There was a problem while creating an environment')
        print(error)
