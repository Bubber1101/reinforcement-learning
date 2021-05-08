import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    scores, q_table = basic_q_learning(number_of_episodes=100, epsilon_decays=False, epsilon=0.8, learning_rate=0.9,
                                       discount_factor=0.98)
    plt.plot(scores)
    plt.show()
    print(q_table)
