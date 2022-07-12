from collections import deque
import numpy as np
import torch

def train_agent(
    env, brain_name, agent, n_episodes=1000, max_t=1000, eps_end=0.01, eps_decay=0.99,
    train_every_n=10, verbose=True
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timetake_actions per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = 1                            # initialize epsilon
    passed = False
    intervals = [5, 10, 20, 50, 100, 200, 500]
    print_every = 1
    for i in intervals[::-1]:
        if n_episodes / i >= 10:
            print_every = i
            break

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            agent.take_action(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
            if (t+1) % train_every_n == 0:
                agent.learn()
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        # print results:
        if verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'\
                .format(i_episode, np.mean(scores_window), eps),
                end="")
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=13 and passed == False:
                print('\n\tAgent passed muster in {:d} episodes!  (Average Score: {:.2f})'.format(i_episode-50, np.mean(scores_window)))
                torch.save(agent.qnetwork_actor.state_dict(), 'base_checkpoint.pth')
                passed = True
                # break

    if passed == True:
        torch.save(agent.qnetwork_actor.state_dict(), 'base_checkpoint_final.pth')

    return scores
