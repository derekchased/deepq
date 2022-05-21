import gym 
import random


env = gym.make("CartPole-v1")
def Random_games():
    for episode in range(10):
        env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(t, next_state, reward, done, info, action)
            if done:
                break
Random_games()