import gymnasium as gym
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial

plt.style.use('ggplot')

env = gym.make("Blackjack-v1", render_mode="human")#初始化環境
env.action_space, env.observation_space

def sample_policy(observation):
    (score, dealer_score, usable_ace), x = observation
    return 0 if score >= 20 else 1

def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    print(observation)
    while True:
        states.append(observation)
        action = policy(observation)
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return states, actions, rewards

def first_visit_mc_prediction(policy, env, n_episodes):
    value_table = defaultdict(float)
    N = defaultdict(int)
    print(type(N))

    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy, env)
        returns = 0
        for t  in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            returns += R
            if S not in states[:t]:
                print("N[S]: " + N[S])
                N[S] += 1
                value_table[S] += (returns - value_table[S] / N[S])
    return value_table

value = first_visit_mc_prediction(sample_policy, env, n_episodes=500000)
print(value)