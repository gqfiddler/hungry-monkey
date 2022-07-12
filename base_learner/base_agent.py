import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on', device, '\n')

class Agent():

    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, fcn_sizes=[64], seed=1):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = GAMMA

        # Q-Network
        self.qnetwork_actor = QNetwork(state_size, action_size, fcn_sizes, seed).to(device)
        self.qnetwork_evaluator = QNetwork(state_size, action_size, fcn_sizes,seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_actor.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.num_steps = 0
        self.mse_loss = torch.nn.MSELoss()
        self.batch_warned=False

    def take_action(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.num_steps += 1

    def select_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = self.qnetwork_actor(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy().squeeze())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, gamma=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if len(self.memory) < BATCH_SIZE:
            if self.batch_warned == False:
                print('Warning: insufficient stored steps; skipping training')
                self.batch_warned = True
            return None
        if gamma is None:
            gamma = self.gamma

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        # epsilon = min(1, max(0.02, 20 / (self.num_steps**0.5)))
        pred_rewards = self.qnetwork_actor(states).gather(1, actions)
        true_rewards = rewards + gamma * self.qnetwork_evaluator(next_states).detach().max(1)[0].unsqueeze(1)
        self.qnetwork_actor.train() # set network to training mode
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_rewards, true_rewards)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_actor, self.qnetwork_evaluator, TAU)
        self.qnetwork_actor.eval() # set actor network back to inference mode


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if len(self.memory) > self.buffer_size:
            self.memory.popleft()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
