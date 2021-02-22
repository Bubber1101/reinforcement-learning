import matplotlib.pyplot as plt
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

plt.style.use('seaborn-whitegrid')

from basic_sarsamax import basic_q_learning


def random_action(path_to_executable):
    unity_env = UnityEnvironment(path_to_executable)
    env = UnityToGymWrapper(unity_env)
    for i_episode in range(20):
        env.reset()
        for t in range(50):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()




if __name__ == '__main__':
    # randomAction("C:/Users/patry/Dev/Thesis/envs/Worm/UnityEnvironment.exe")
    # random_action("C:/Users/patry/Dev/Thesis/envs/Basic/Basic.exe")
    scores = basic_q_learning(number_of_episodes=100, epsilon_decays=False, epsilon=0.8, learning_rate=0.9,
                              discount_factor=0.98)
    plt.plot(scores)
    plt.show()
