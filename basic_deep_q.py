from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def basic_build_model():
    pass


def get_env():
    """
    Creates an Unity environment for `Basic <https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning
    -Environment-Examples.md#basic>`_ example
    :returns: UnityToGymWrapper
    """
    unity_env = UnityEnvironment("C:/Users/patry/Dev/Thesis/envs/Basic/Basic.exe", no_graphics=False)
    return UnityToGymWrapper(unity_env)


def basic_deep_q_learning():
    env = get_env()
    model = basic_build_model()


if __name__ == '__main__':
    results = basic_deep_q_learning()
