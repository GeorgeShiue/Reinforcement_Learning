import gymnasium as gym
env = gym.make("BipedalWalker-v3", render_mode="human")

for _ in range(100):
    observation = env.reset()
    for i in range(1600):
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if i % 10 == 0:
            print("{} timestemps taken for the episode".format(i+1))
        if terminated or truncated:
            print("{} timestemps taken for the episode".format(i+1))
            break

env.close()