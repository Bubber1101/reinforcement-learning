from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from tensorflow.python.keras.layers import Activation, Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam


def basic_build_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())
    return model


def get_env() -> UnityToGymWrapper:
    """
    Creates an Unity environment for `Basic <https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning
    -Environment-Examples.md#basic>`_ example
    """
    unity_env = UnityEnvironment("C:/Users/patry/Dev/Thesis/envs/Basic/Basic.exe", no_graphics=False)
    return UnityToGymWrapper(unity_env)


def basic_deep_q_learning():
    env = get_env()
    model = basic_build_model(env)
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=1000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=600, visualize=False, verbose=2)
    env.reset()
    # dqn.save_weights("weights.xd", True)
    dqn.test(env, nb_episodes=5, visualize=True, verbose=2)


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
if __name__ == '__main__':
    results = basic_deep_q_learning()
