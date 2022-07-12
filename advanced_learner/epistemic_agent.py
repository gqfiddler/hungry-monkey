import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork


# BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
# GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
STEPS_TILL_EPS = 5000
STEPS_TILL_NOT_EPS = 40000
# STEPS_TILL_EPS = 35000
# STEPS_TILL_NOT_EPS = 180000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on', device, '\n')


class Agent():

    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_size, replay_epsilon, n_multi_step=1, gamma=0.99,
        buffer_size=int(2e5), prioritize_replay=True, gamma_inv_decay=None,
        use_max_next=True, fcn_sizes=[64,64,64,64], truncate_by_min_error=False,
        soft_update=False, ep_eps=0.5, ep_eps_min=0.01, seed=1
    ):
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
        self.gamma = gamma
        self.n_multi_step = n_multi_step
        self.use_max_next = use_max_next
        self.gamma_inv_decay = gamma_inv_decay
        self.soft_update = soft_update
        self.ep_eps = ep_eps
        self.ep_eps_min = ep_eps_min
        if gamma_inv_decay is not None:
            self.gamma = 0.5

        # Q-Network
        self.qnetwork_actor = QNetwork(state_size, action_size, fcn_sizes, seed).to(device)
        self.qnetwork_evaluator = QNetwork(state_size, action_size, fcn_sizes, seed).to(device)
        self.epistemic_estimator = QNetwork(state_size, action_size, fcn_sizes, seed).to(device)
        self.qnetwork_actor.eval()
        self.qnetwork_evaluator.eval()
        self.epistemic_estimator.eval()
        self.act_optimizer = optim.Adam(self.qnetwork_actor.parameters(), lr=LR)
        self.eval_optimizer = optim.Adam(self.qnetwork_evaluator.parameters(), lr=LR)
        self.epistemic_errors = []
        self.batch_warned=False

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, buffer_size, BATCH_SIZE, replay_epsilon, n_multi_step,
            prioritize_replay, truncate_by_min_error, seed
        )
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.num_steps = 0
        self.mse_loss = torch.nn.MSELoss()

    def record_action(self, state, action, reward, pred_reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, pred_reward, next_state, done, self.gamma)
        self.num_steps += 1
        if self.num_steps == STEPS_TILL_EPS:
            print('\nnum_steps reached STEPS_TILL_EPS!\n')
        if self.num_steps == STEPS_TILL_NOT_EPS:
            print('\nnum_steps reached STEPS_TILL_NOT_EPS!\n')
        if self.gamma_inv_decay is not None:
            self.gamma *= self.gamma_inv_decay

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
            action_values = self.qnetwork_actor(state).cpu().data.numpy().squeeze()
            error_estimates = self.epistemic_estimator(state).cpu().data.numpy().squeeze()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values), max(action_values)
        else:
            if random.random() < self.ep_eps and self.num_steps >= STEPS_TILL_EPS \
                and self.num_steps <= STEPS_TILL_NOT_EPS:
                choice_idx = np.argmax(error_estimates)
                return choice_idx, action_values[choice_idx]
            else:
                choice_idx = random.choice(np.arange(self.action_size))
                return choice_idx, action_values[choice_idx]

    def learn(self, learner_net='learner', use_max_next=True, gamma=None):
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
        if learner_net=='learner':
            learner_net, evaluator_net = self.qnetwork_actor, self.qnetwork_evaluator
            optimizer = self.act_optimizer
        elif learner_net=='evaluator':
            learner_net, evaluator_net = self.qnetwork_evaluator, self.qnetwork_actor
            optimizer = self.eval_optimizer
        else:
            raise ValueError("learner_net parameter must be set to either 'learner' or 'evaluator'")

        idx, experience_data = self.memory.sample()
        states, actions, rewards, next_states, dones = experience_data

        ## TODO: compute and minimize the loss
        # epsilon = min(1, max(0.02, 20 / (self.num_steps**0.5)))
        pred_rewards = learner_net(states).gather(1, actions)
        if self.use_max_next:
            next_expected_reward = evaluator_net(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            action_values = learner_net(next_states).detach()
            next_actions = np.argmax(action_values, axis=1).cuda()
            next_expected_reward = evaluator_net(next_states).detach().gather(1, next_actions.unsqueeze(1))
        # print('rewards:', rewards.shape)
        # print('actions:', actions.shape)
        # print('next_expected_reward:', next_expected_reward.shape)
        true_rewards = rewards + (gamma**self.n_multi_step) * next_expected_reward * (1 - dones)
        # print('true_rewards:', true_rewards.shape)
        # print('pred_rewards:', pred_rewards.shape)
        learner_net.train() # set network to training mode
        optimizer.zero_grad()
        loss = self.mse_loss(pred_rewards, true_rewards)
        loss.backward()
        optimizer.step()
        self.memory.update_errors(
            idx,
            true_rewards.cpu().numpy().squeeze(),
            pred_rewards.cpu().detach().numpy().squeeze()
        )
        learner_net.eval() # set actor network back to inference mode

        # train epistemic estimator
        if self.num_steps >= STEPS_TILL_EPS//2 and self.num_steps <= STEPS_TILL_NOT_EPS:
            optimizer.zero_grad()
            self.epistemic_estimator.train()
            pred_errors = self.epistemic_estimator(states).gather(1, actions)
            real_errors = (true_rewards.detach() - pred_rewards.detach())**2
            error_loss = self.mse_loss(pred_errors, real_errors)
            error_loss.backward()
            optimizer.step()
            self.epistemic_estimator.eval()
            self.epistemic_errors.append(error_loss.detach().cpu().numpy())


        if self.soft_update:
            # ------------------- update target network ------------------- #
            self.soft_update(learner_net, evaluator_net, TAU)
            evaluator_net.eval() # set actor network back to inference mode

    def clear_cache(self):
        self.memory.clear_cache()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): model from which weights will be copied
            target_model (PyTorch model): model to which weights will be copied
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, action_size, buffer_size, batch_size, replay_epsilon, n_multi_step,
        prioritize_replay=True, truncate_by_min_error=False, seed=1
    ):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_epsilon = replay_epsilon
        self.n_multi_step = n_multi_step
        self.experience_cache = deque(maxlen=self.n_multi_step)
        self.memory = []
        self.memory_idx = [] # just a list of range(len(memory)) to avoid regenerating that every time we sample
        self.errors = []
        self.prioritize_replay = prioritize_replay
        self.mean_error = 1000 # start high --> buffer sampling is initially random
        self.memory_truncated = False
        self.truncate_by_min_error = truncate_by_min_error
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "pred_reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)

    def clear_cache(self):
        self.experience_cache = deque(maxlen=self.n_multi_step)

    def add(self, state, action, reward, pred_reward, next_state, done, gamma):
        """Add a new experience to memory."""

        e_new = self.experience(state, action, reward, pred_reward, next_state, done)

        # case where experience cache not yet full, just add experience to cache
        if not done and len(self.experience_cache) < self.n_multi_step:
            self.experience_cache.append(e_new)
            return None

        # case where done, move ALL experiences in cache to memory
        if done:
            num_experiences_to_memory = len(self.experience_cache)
        # USUAL CASE: where not done and experience cache is full
        else:
            num_experiences_to_memory = 1

        for n in range(num_experiences_to_memory):
            # calculate full cumulative reward of the OLDEST entry in cache
            # = discounted sum of:
            #   * empirical single-step rewards (except the newest cached step)
            #   * predicted cumulative reward of the newest step
            cached_rewards = [m.reward for m in self.experience_cache]
            rewards = cached_rewards + (done * pred_reward) # newest reward is 0 if done
            discounted_rewards = [r*(gamma**i) for i, r in enumerate(rewards)]
            empirical_reward = sum(discounted_rewards[:-1])
            full_reward = sum(discounted_rewards)
            pred_full_reward = self.experience_cache[0].pred_reward
            # update the oldest cached memory's reward and error,
            # then store in memory
            old_e_updated = self.experience(
                self.experience_cache[0].state, self.experience_cache[0].action,
                empirical_reward, self.experience_cache[0].pred_reward,
                next_state, done)
            self.memory.append(old_e_updated)
            self.memory_idx.append(len(self.memory_idx))
            self.errors.append(abs(full_reward-pred_full_reward)*2) # *2 to prioritize not-yet-learned experiences
            if done:
                _ = self.experience_cache.popleft()
        if done:
            # put the current experience in memory as is – no future rewards to add
            self.memory.append(e_new)
            self.memory_idx.append(len(self.memory_idx))
            self.errors.append(abs(reward-pred_reward)*2) # *2 to prioritize not-yet-learned experiences
        else:
            self.experience_cache.append(e_new) # automatically bumps oldest experience out

        # do batch deletions for sake of efficiency
        if len(self.memory) - self.buffer_size > 500:
            if not self.memory_truncated:
                print('\nWARNING: exceeded memory buffer; truncating memory henceforth\n')
                self.memory_truncated = True
            if self.truncate_by_min_error == False:
                del self.errors[:500]
                del self.memory_idx[-500:]
                del self.memory[:500]
            else:
                del self.memory_idx[-500:]
                del_idx = sorted(list(np.argsort(self.errors)[:500]))
                for i in del_idx[::-1]:
                    del self.errors[i]
                    del self.memory[i]


    def sample(self):
        """Sample a batch of experiences from memory."""


        if (len(self.memory) + 1) % 500 == 0:
            self.mean_error = np.mean(self.errors)

        # scale down weights to avoid too large of numbers when random.choices()
        # calculates cumulative weights
        if self.prioritize_replay:
            weights = (np.array(self.errors) + (self.mean_error * self.replay_epsilon)) / len(self.errors)
        else:
            weights = None
        idx = random.choices(
            self.memory_idx,
            k=self.batch_size,
            weights=weights
        )
        experiences = [self.memory[i] for i in idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (idx, (states, actions, rewards, next_states, dones))

    def update_errors(self, idx, true_rewards, pred_rewards, soft_update=False):
        # update errors after learning
        errors = abs(true_rewards - pred_rewards)
        for i, error in enumerate(errors):
            if soft_update:
                self.errors[idx[i]] = (self.errors[idx[i]] + error) / 2
            else:
                self.errors[idx[i]] = error

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
