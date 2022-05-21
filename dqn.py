import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        self.epsilon = 1.0

        # self.fc1 = nn.Linear(4, 256)
        # self.fc2 = nn.Linear(256, self.n_actions)
        #
        # self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        prob = np.random.uniform(0, 1)
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - (self.eps_start - self.eps_end)/self.anneal_length
        if prob >= self.epsilon and exploit == True:
            state_action_pair = self.forward(observation.to(device))  # Q(s, left) and Q(s, right) left:0, right:1
            with torch.no_grad():
                return state_action_pair.max(1)[1].view(1, 1)
        else:

            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        raise NotImplmentedError

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    (obs, action, next_obs, reward) = memory.sample(dqn.batch_size)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            next_obs)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in next_obs if s is not None])
    state_batch = torch.cat(obs).to(device)
    action_batch = torch.cat(action).to(device)
    reward_batch = torch.cat(reward).to(device)
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    state_action_pair = dqn(state_batch).gather(1, action_batch.unsqueeze(1))
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    next_state_values = torch.zeros(dqn.batch_size, device=device)
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * dqn.gamma) + reward_batch
    # Compute loss.
    q_values = state_action_pair
    q_value_targets = expected_state_action_values
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    # for param in dqn.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()
