import gymnasium as gym
import numpy as np

def output(table):
    for i in range(len(table)):
        print(table[i], end = ' ')
        if (i + 1) % 4 == 0:
            print()
    print()

def compute_value_function(policy, gamma = 1.0):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-10
    j = 0
    while True:
        j += 1
        updated_value_table = np.copy(value_table)
        #找出所有狀態下對應到的隨機策略中的動作的Q值，用來更新每個狀態的價值函數
        for state in range(env.observation_space.n):
            action = policy[state]
            next_state_reward = []
            #加總從某狀態轉移到剩下所有狀態的獎勵
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                #套用Q方程式：Q(s,a) = 轉移機率 * (獎勵機率 + gamma * 下一個狀態的價值)
                next_state_reward.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
            value_table[state] = np.sum(next_state_reward) #注意這裡和書中不同
        
        #價值函數若收斂到一定範圍就結束迴圈
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print(j)
            break
    return value_table
    
#從最佳價值函式中找出最佳策略
def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
                #print(trans_prob, next_state, reward_prob)
        print(Q_table)
        policy[state] = np.argmax(Q_table)
    return policy

def policy_iteration(env, gamma = 1.0):
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    gamma = 1.0
    
    for i in range(no_of_iterations):
        print(i)
        #從隨機策略中找出最佳價值函數
        new_value_function = compute_value_function(random_policy, gamma)
        output(new_value_function)
        #從最佳價值函數中找出最佳策略
        new_policy = extract_policy(new_value_function, gamma)
        output(new_policy)

        if np.all(random_policy == new_policy):
            print('Policy-Iteration converged at step {}.'.format(i + 1))
            break
        #將下一輪的隨機策略設定為本輪的最佳策略
        random_policy = new_policy
    return new_policy

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
#print(policy_iteration(env))
output(policy_iteration(env))