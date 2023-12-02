import gymnasium as gym
import numpy as np

def value_iteration(env, gamma = 1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_interations = 100000
    threshold = 1e-20

    for i in range(no_of_interations):
        update_value_table = np.copy(value_table) 
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_state_reward = []
                for next_str in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_str
                    #下面一行套用Q方程式：Q(s,a) = 轉移機率 * (獎勵機率 + gamma * 下一個狀態的價值)
                    #print(trans_prob * (reward_prob + gamma * update_value_table[next_state]))
                    next_state_reward.append((trans_prob * (reward_prob + gamma * update_value_table[next_state])))
                Q_value.append(np.sum(next_state_reward))#放入某狀態的Q值總和
            value_table[state] = max(Q_value) #用最大的Q值更新狀態值

        #下面若收斂到一定範圍就結束迴圈
        if(np.sum(np.fabs(update_value_table - value_table)) <= threshold): 
            print('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return value_table

#下面函式找出最佳價值函式中的最佳策略
def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_str in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_str
                #print(value_table[next_state])
                #print((trans_prob * (reward_prob + gamma * value_table[next_state])))
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
print(env.observation_space.n)

optimal_value_function = value_iteration(env = env, gamma = 1.0)
print(optimal_value_function)
optimal_policy = extract_policy(optimal_value_function, gamma = 1.0)
print(optimal_policy)