import argparse

import matplotlib.pyplot as plt
import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException


def random_action(path_to_executable, number_of_episodes):
    """
    This method runs an environment and performs random actions on it until a reward is reached
     for a given number of episodes

    :param path_to_executable: Path to an executable that contains a unity environment
    :param number_of_episodes: Number of episodes the policy should be run for,
           episode ends when the environment returns Done as true
    :return: List of scores per each episode
    """
    try:
        unity_env = UnityEnvironment(path_to_executable)
        env = UnityToGymWrapper(unity_env)
    except UnityEnvironmentException as error:
        print('There was a problem while creating an environment')
        print(error)

    scores = []

    for episode in range(number_of_episodes):
        env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score = score + reward
        scores.append(score)
    env.close()
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script takes any unity environment and performs'
                                                 ' random actions for given number of episodes')
    parser.add_argument('-f', '--file',
                        dest='filepath', required=True,
                        help='Executable file containing for a unity environment')
    parser.add_argument('-e', '--episodes',
                        dest='episodes', required=False, default=10,
                        help='Number of episodes that should be run with random action policy')
    args = parser.parse_args()

    scores = 0
    scores = random_action(args.filepath, args.episodes)

