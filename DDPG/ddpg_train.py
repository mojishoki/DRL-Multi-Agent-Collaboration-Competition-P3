import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque 
from hyperparameters import *
from itertools import cycle

def ddpg_train(agents,env,brain_name,n_episodes=1000, max_t=2000,target_score=0.5,window_size=100, print_every=10, train_mode=True):
    """Deep Deterministic Policy Gradient (DDPG)
    Params
    ======
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        window_size (int)     : moving average is taken over window_size episodes
        target_score (float)  : min moving average score to reach to solve env
        print_every (int)     : print results print_every episode
        train_mode (bool)     : if `True` set environment to training mode, if `False` watch it play while training
    """
    stats=f'Checkpoints/'
    mean_scores,min_scores,max_scores,moving_avgs=([] for _ in range(4))
    stats+=f'{agents[0].batch_size}{agents[0].buffer_size}{agents[0].gamma}{agents[0].tau}{agents[0].lr_actor}{agents[0].lr_critic}{agents[0].weight_decay}{agents[0].random_seed}'
    stats+=f'{agents[0].update_every}{agents[0].learn_num}'
    stats+=f'{agents[0].noise.sigma}{agents[0].noise.theta}'
    scores_window = deque(maxlen=window_size)  # mean scores from most recent episodes
    durations=[]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset environment
        states = env_info.vector_observations                   # get current state for each agent      
        scores = np.zeros(len(env_info.agents))                 # initialize score for each agent
        for agent in agents:
            agent.reset()                                           # reset the OU noise
        start_time = time.time()
        for t in range(max_t):
            actions=[]
            for state,agent in zip(states,cycle(agents)):
                actions.append(agent.act(state[np.newaxis,:], add_noise=True))        # select an action
            env_info = env.step(actions)[brain_name]            # send actions to environment
            next_states = env_info.vector_observations          # get next state
            rewards = env_info.rewards                          # get reward
            dones = env_info.local_done                         # see if episode has finished
            # save experience to replay buffer, perform learning step at defined interval
            for state, action, reward, next_state, done,agent in zip(states, actions, rewards, next_states, dones,cycle(agents)):
                agent.step(state, action, reward, next_state, done)             
            states = next_states
            scores += rewards
            if np.any(dones):                                   # exit loop when episode ends
                break
        duration = time.time() - start_time
        durations.append(duration)
        min_scores.append(np.min(scores))             
        max_scores.append(np.max(scores))                     
        mean_scores.append(np.mean(scores))           
        scores_window.append(max_scores[-1])         
        moving_avgs.append(np.mean(scores_window))    # save moving average
        print('\rEpisode {} ({} sec)  -- \tMin: {:.3f}\tMax: {:.3f}\tMean: {:.3f}\tMov. Avg: {:.3f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]),end="")
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.3f}\tMax: {:.3f}\tMean: {:.3f}\tMov. Avg: {:.3f}'.format(\
                  i_episode, round(sum(durations[-10:])), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))
                  
        if moving_avgs[-1] >= target_score and i_episode >= window_size:
            print('\nEnvironment SOLVED in {} episodes and {} time!\tMoving Average ={:.3f} over last {} episodes'.format(\
                                    i_episode, round(sum(durations)), moving_avgs[-1], window_size))            
            if train_mode:
                stats+=f'solved{i_episode}'
                for i in range(len(agents)):
                    torch.save(agents[i].actor_local.state_dict(), stats+f'actor_ckpt{i}'+f'.pth')
                    torch.save(agents[i].critic_local.state_dict(), stats+f'critic_ckpt{i}'+f'.pth')
                np.savetxt(stats+f'mean',mean_scores)
                np.savetxt(stats+f'min',min_scores)
                np.savetxt(stats+f'max',max_scores)
                np.savetxt(stats+f'avg',moving_avgs)
                np.savetxt(stats+f'durations',durations)
            break
            
    return mean_scores,min_scores,max_scores,moving_avgs,durations,stats