import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")#初始化環境
#env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("CarRacing-v2", render_mode="human")

observation, info = env.reset()#獲得初始觀測值和環境資訊

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)#將主體動作後的結果導入觀察、獎勵等等

    if terminated or truncated:#若主體在達成目標成功或失敗了，或是在一定動作數量之後
        observation, info = env.reset()#重置環境

env.close()#關閉環境