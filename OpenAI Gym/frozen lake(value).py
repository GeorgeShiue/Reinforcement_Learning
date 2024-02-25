import gymnasium as gym
import numpy as np

def output(table):
    for i in range(len(table)):
        print(table[i], end = ' ')
        if (i + 1) % 4 == 0:
            print()
    print()

#找出考量所有狀態下的各種動作的最佳價值函數
def value_iteration(env, gamma = 1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_interations = 100000
    threshold = 1e-20

    for i in range(no_of_interations):
        update_value_table = np.copy(value_table)
        #找出所有狀態下的各種動作的最大Q值，用來更新每個狀態的價值函數
        for state in range(env.observation_space.n):
            Q_value = []
            #找出當前狀態下的各種動作的Q值，並取出最大值用來更新當前狀態的價值函數
            for action in range(env.action_space.n):
                next_state_reward = []
                #加總從某狀態轉移到剩下所有狀態的獎勵
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    #套用Q方程式：Q(s,a) = 轉移機率 * (獎勵機率 + gamma * 下一個狀態的價值)
                    next_state_reward.append((trans_prob * (reward_prob + gamma * update_value_table[next_state])))
                Q_value.append(np.sum(next_state_reward))
            value_table[state] = max(Q_value)

        output(value_table)

        #價值函數若收斂到一定範圍就結束迴圈
        if(np.sum(np.fabs(update_value_table - value_table)) <= threshold): 
            print('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return value_table

#從最佳價值函式中找出最佳策略
def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    #找出每個狀態下的最佳動作
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        #找出當前狀態下的最佳動作
        for action in range(env.action_space.n):
            #加總轉移狀態時執行某動作的Q值
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                #套用Q方程式：Q(s,a) = 轉移機率 * (獎勵機率 + gamma * 下一個狀態的價值)
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        print(Q_table)
        print(np.argmax(Q_table))
        policy[state] = np.argmax(Q_table)
    return policy

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

optimal_value_function = value_iteration(env = env, gamma = 1.0)
#print(optimal_value_function)
output(optimal_value_function)
optimal_policy = extract_policy(optimal_value_function, gamma = 1.0)
#print(optimal_policy)
output(optimal_policy)