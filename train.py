import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    
    # TODO: Create and initialize target Q-network.
    dqn_target = DQN(env_config=env_config).to(device)
    # dqn_target = copy.deepcopy(dqn)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")


    for episode in range(env_config['n_episodes']):
        print(f"episode {episode}")
        done = False

        # [-0.03674591 -0.04221547 -0.04493564  0.00675605]
        curr_obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        
        # for cart pole the the observation is something
        # like env.reset() = [-0.03674591 -0.04221547 -0.04493564  0.00675605]
        step_counter = 1
        while not done:
            # print(f"step")

            # TODO: Get action from DQN.
            action = dqn.act(curr_obs)
            
            # Act in the true environment.
            next_obs, reward, done, info = env.step(action.item())


            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
            # print(f"next_obs {next_obs}")

            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            print(curr_obs, action, next_obs, reward)
            memory.push(curr_obs, action, next_obs, reward)

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if env_config["train_frequency"]%step_counter == 0:
                optimize(dqn, dqn_target, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if env_config["target_update_frequency"]%step_counter == 0:
                dqn_target.load_state_dict(dqn.state_dict())

            step_counter += 1

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
