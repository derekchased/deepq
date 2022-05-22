import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'], default='Pong-v0')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'Pong-v0': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    obs_stack_size = 4
    num_step = 0
    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0)
        # obs_stack /= 255.0
        loss_ = []
        while not done:
            # TODO: Get action from DQN.
            action = dqn.act(obs_stack, exploit=True)

            # Act in the true environment.
            obs_next, reward, done, info = env.step(action.item())
            # Preprocess incoming observation.
            if not done:
                obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1)
                # next_obs_stack /= 255.0
            else:
                obs_next = None
                next_obs_stack = None
            memory.push(obs_stack, torch.asarray([action], device=device), next_obs_stack, torch.asarray([reward], device=device))
            obs = obs_next
            obs_stack = next_obs_stack

            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            # if done:
            num_step += 1
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if num_step % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer)
                loss_.append(loss)
            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if num_step % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())
        # print(episode, loss_)




        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')

    # Close environment after training is completed.
    env.close()
