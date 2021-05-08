import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


def basic_q_learning(discount_factor=0.95,
                     learning_rate=0.1,
                     epsilon=1,
                     epsilon_decay=0.5,
                     epsilon_min=0.01,
                     number_of_episodes=100,
                     epsilon_decays=True):
    """
        This performs a sarsamax training on environment "Basic" located under "envs/Basic.exe".
        The environment was prepared in unity and this script uses
        Unity ML-Agents Python Interface(https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs)
        in order to communicate with the executable.
    """

    unity_env = UnityEnvironment("envs/Basic.exe", no_graphics=False)
    env = UnityToGymWrapper(unity_env)
    possible_states = env.observation_space.shape[0]
    possible_actions = env.action_space.n
    q_table = np.zeros((possible_states, possible_actions))
    scores = np.zeros((number_of_episodes, 1))

    for i_episode in range(number_of_episodes):
        if epsilon_decays:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
        score = 0
        state = env.reset()
        state = np.where(state == 1)[0][0]
        done = False
        while not done:
            action = choose_explore_exploit(q_table, state, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action)
            if next_state is not None:
                next_state = np.where(next_state == 1)[0][0]
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
        scores[i_episode] = score

    return scores, q_table


def update_q(q_table,
             reward,
             learning_rate,
             discount_factor,
             current_state,
             current_action,
             next_state):
    next_q = get_next_q(next_state, q_table)

    current_q = q_table[current_state][current_action]
    return current_q + learning_rate * (reward + discount_factor * next_q - current_q)


def get_next_q(next_state, q_table):
    if next_state is None:
        return 0
    else:
        next_action = np.argmax(q_table[next_state])
        return q_table[next_state][next_action]


def choose_explore_exploit(q_table, state, epsilon, action_space: UnityToGymWrapper.action_space):
    rand = np.random.rand()
    if rand < epsilon:
        return action_space.sample()
    else:
        return np.argmax(q_table[state])


if __name__ == '__main__':
    basic_q_learning()
