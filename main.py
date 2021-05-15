import matplotlib.pyplot as plt
import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

from basic_sarsamax import basic_q_learning

plt.style.use('seaborn-whitegrid')


def random_action(path_to_executable):
    unity_env = UnityEnvironment(path_to_executable)
    env = UnityToGymWrapper(unity_env)
    for i_episode in range(20):
        env.reset()
        for t in range(20):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def single_action(path, action):
    unity_env = UnityEnvironment(path)
    env = UnityToGymWrapper(unity_env)
    env.reset()
    for t in range(20):
        env.render()
        observation, reward, done, info = env.step(action)
        print(str(reward) + " and " + str(observation))
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    env.close()


def move_left():
    single_action("envs/Basic.exe", 1)


def move_right():
    single_action("envs/Basic.exe", 2)


def epsilon_plot(beggining, decay, min, x):
    values = np.zeros(x)
    epsilon = beggining

    for i in range(x):
        epsilon = max(epsilon * decay, min)
        values[i] = epsilon
    return values


if __name__ == '__main__':
    # scores, q_table = basic_q_learning(number_of_episodes=100, epsilon_decays=False, epsilon=0.8, learning_rate=0.9,
    #                                    discount_factor=0.98)
    scores, q_table = basic_q_learning(number_of_episodes=100)

    regression_line = np.poly1d(np.polyfit(np.arange(start=0, stop=100), scores, 5))
    plt.plot(scores, label='Episode Score')
    plt.plot(np.arange(start=0, stop=100), regression_line(np.arange(start=0, stop=100)), label='Regression')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right")

    plt.show()
    plt.plot(scores, label='Episode Score')
    plt.plot(epsilon_plot(1, 0.75, 0.01, 100), '-r', label='Epsilon value')
    plt.ylim(-0.5, 1.1)
    plt.legend(loc="lower right")

    plt.show()

    print(q_table)
