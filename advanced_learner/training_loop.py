from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt


def train_agent(
    env, brain_name, agent, n_episodes=1000, max_t=1000, eps_end=0.01, eps_decay=0.995,
    train_every_n=10, double_dqn=False, n_episodes_bailout=None, bailout_threshold=1,
    print_every=None, is_epistemic=False, verbose=True
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
    max_score = 0

    # figure out how often to print outputs
    intervals = [5, 10, 20, 50, 100, 200, 500]
    if print_every is None:
        print_every = 1
        for i in intervals[::-1]:
            if n_episodes / i >= 10:
                print_every = i
                break

    # run main training loop
    for i_episode in range(1, n_episodes+1):
        agent.clear_cache()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action, pred_reward = agent.select_action(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            agent.record_action(state, action, reward, pred_reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
            if (t+1) % train_every_n == 0:
                agent.learn(learner_net='learner')
                if double_dqn:
                    agent.learn(learner_net='evaluator')

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        # print results:
        if verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'\
                .format(i_episode, np.mean(scores_window), eps),
                end="")
            if i_episode % print_every == 0:
                print()
            current_score = np.mean(scores_window)
            if current_score >= 13:
                if passed == False:
                    print('\n\tAgent passed muster in {:d} episodes!  (Average Score: {:.2f})'.format(i_episode-50, np.mean(scores_window)))
                    torch.save(agent.qnetwork_actor.state_dict(), 'ep_checkpoint.pth')
                    passed = True
                    # break
                if current_score > max_score and i_episode % 10 == 0:
                    max_score = torch.save(agent.qnetwork_actor.state_dict(), 'adv_checkpoint.pth')
                    max_score = current_score
            if n_episodes_bailout is not None \
            and i_episode >= n_episodes_bailout \
            and current_score < bailout_threshold:
                print('score below threshold; bailing out...')
                break

    if passed == True:
        torch.save(agent.qnetwork_actor.state_dict(), 'adv_checkpoint.pth')

    if is_epistemic == True:
        errors = np.convolve(agent.epistemic_errors, np.ones(100)/100, mode='valid')
        plt.plot(errors)
        plt.xlabel('training step')
        plt.ylabel('error-estimation error')
        plt.show()

    return scores
