import argparse
import os
from argparse import ArgumentParser

import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def basic_q_learning(
        filepath,
        discount_factor=0.95,
        learning_rate=0.1,
        epsilon=1,
        epsilon_decay=0.75,
        epsilon_min=0.01,
        number_of_episodes=100,
        epsilon_decays=True):
    """
    This performs a sarsamax training on environment "Basic" located under "envs/Basic.exe".
    The environment was prepared in unity and the script uses
    Unity ML-Agents Python Interface(https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs)
    in order to communicate with the executable.

    :param filepath:
    :param discount_factor:
    :param learning_rate:
    :param epsilon:
    :param epsilon_decay:
    :param epsilon_min:
    :param number_of_episodes: number of episodes the agent should learn
    :param epsilon_decays: a boolean specifying if the epsilon value should decrease with each episode
    :return:
    """
    env = get_env(filepath)
    possible_states = env.observation_space.shape[0]
    possible_actions = env.action_space.n
    q_table = np.zeros((possible_states, possible_actions))
    scores = []

    for i_episode in range(number_of_episodes):
        if epsilon_decays:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
        score = 0
        state = env.reset().argmax()
        done = False
        while not done:
            action = choose_explore_exploit(q_table, state, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action)
            if next_state is not None:
                next_state = next_state.argmax()
            score = score + reward
            q_table[state][action] = update_q(q_table,
                                              reward,
                                              learning_rate,
                                              discount_factor,
                                              state,
                                              action,
                                              next_state)
            if not done:
                state = next_state
        scores.append(score)

    return scores, q_table


def update_q(q_table,
             reward,
             learning_rate,
             discount_factor,
             current_state,
             current_action,
             next_state):
    expected_next_q_value = get_max_q_value_for_state(next_state, q_table)
    current_q_value = q_table[current_state][current_action]

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
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError("{0} is not a path to a file".format(filepath))
    return filepath


def get_env(filepath):
    unity_env = UnityEnvironment(filepath, no_graphics=False)
    return UnityToGymWrapper(unity_env)


if __name__ == '__main__':
    parser = ArgumentParser(description="Sarsamax implementation on 'Basic' unity environment")
    parser.add_argument("-f", "--file",
                        dest="filepath", required=True, type=existing_file,
                        help="Executable file containing a version of 'Basic' environment from ./envs folder")
    args = parser.parse_args()
    basic_q_learning(args.filepath)
