import numpy as np
import torch
import time 
from collections import deque
from configs import *

def MADDPG_train(agent,env,brain_name,pretraining=False,action_size=ACTION_SIZE,ratio=RATIO,hard_end_noise=True,
               hard_end=HARD_END, noise=NOISE,noise_end=NOISE_END,n_episodes=20000, max_t=2000, target_score=0.5,
               window_size=100, print_every=100, train_mode=True, alter_factor=ALTER_FACTOR):
    """
    Params
    ======
        noise (float)         : OU noise scale, default at 5.
        hard_end_noise (bool) : to set the OU noise scale at a minimum given by noise_end (default 0.1) after episode hard_end (default 300)
        ratio (float)         : the ratio with which the replay buffer is to be pre-filled, default 0.5
        pretraining (bool)    : boost the replay buffer
        alter_factor (float)  : scaling the rewards, default set at 1
        n_episodes (int)      : maximum number of training episodes
        max_t (int)           : maximum number of timesteps per episode
        window_size (int)     : moving average is taken over window_size episodes
        target_score (float)  : min moving average score to reach to solve env
        print_every (int)     : print results print_every episode
        train_mode (bool)     : if `True` set environment to training mode, if `False` watch it play while training
    """
    pretrain_steps = 0
    stats = f'Checkpoints/'
    mean_scores,min_scores,max_scores,moving_avgs = ([] for _ in range(4))
    stats+=f'{agent.batch_size}{agent.buffer_size}{agent.discount_factor}{agent.tau}{agent.lr_actor}{agent.lr_critic}'
    stats=f'{agent.hidden_in_actor}{agent.hidden_out_actor}{agent.hidden_in_critic}{agent.hidden_out_critic1}{agent.hidden_out_critic2}'
    stats+=f'{agent.update_every}{agent.learn_num}'
    scores_window = deque(maxlen=window_size)  # max scores from most recent episodes
    if pretraining:
        pos=0
        neg=0
        neut=0
        while True:                                     
            env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
            obs = env_info.vector_observations                  # get the current state (for each agent)
            scores = np.zeros(len(env_info.agents)) 
            step=0
            if pretrain_steps > agent.buffer_size*ratio: #that's when it is pre-filled
                break
            while True:
                action = np.random.randn(len(env_info.agents), action_size) # select an action (for each agent)
                action = np.clip(action, -1, 1)                  # all actions between -1 and 1
                env_info = env.step(action)[brain_name]           # send all actions to tne environment
                next_obs = env_info.vector_observations         # get next state (for each agent)
                reward = env_info.rewards                         # get reward (for each agent)
                done = env_info.local_done                        # see if episode finished
                reward=[reward*alter_factor for reward in reward] #altering the rewards
                scores += reward                         # update the score (for each agent)
                obs = next_obs                               # roll over states to next time step
                #case-checking whether it is neutral or positive or negative and storing accordingly
                if np.mean(reward)>0. and pos/(pretrain_steps+1e-6)<0.15:
                    pretrain_steps+=1
                    agent.memory.add(obs, action, reward, next_obs, done)
                    pos= pos+1
                elif np.mean(reward)<0. and neg/(pretrain_steps+1e-6)<0.425:
                    pretrain_steps+=1
                    agent.memory.add(obs, action, reward, next_obs, done)
                    neg= neg+1
                elif np.mean(reward)==0. and neut/(pretrain_steps+1e-6)<0.425:
                    pretrain_steps+=1
                    agent.memory.add(obs, action, reward, next_obs, done)
                    neut= neut+1   
                step+=1
                if np.any(done):                                  # exit loop if episode finished
                    break
        print("pretrain_steps : {}, neg : {}, neut : {}, pos : {}" .format(pretrain_steps,neg,neut,pos))   
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset environment
        obs = env_info.vector_observations                   # get current state for each agent      
        scores = np.zeros(len(env_info.agents))                 # initialize score for each agent
#         agent.reset()                                           # reset the OU noise
        start_time = time.time()
        for t in range(max_t):
            action=agent.act(obs,noise)       # select an action
            env_info = env.step(np.array(action).astype(int))[brain_name]            # send actions to environment
            next_obs = env_info.vector_observations          # get next state
            reward = env_info.rewards                          # get reward
            done = env_info.local_done                         # see if episode has finished
            agent.step(obs, action, reward, next_obs, done)             
            obs = next_obs
            reward=[reward*alter_factor for reward in reward]  #scale rewards
            scores += reward
            if hard_end_noise:                                 #checking whether to set OUNoise scale to noise_end
                if i_episode>hard_end: 
                    noise=noise_end
                else:
                    noise=noise-1/(hard_end*agent.update_every)
            if np.any(done):                                   # exit loop when episode ends
#                 if i_episode%print_every==0:  # to check the gauss scale noise uncomment below
#                     print("Gauss scale",agent.maddpg_agent[0].noise.noise(agent.noise_step))
                break
        
        duration = time.time() - start_time
        min_scores.append(np.min(scores))             
        max_scores.append(np.max(scores))                     
        mean_scores.append(np.mean(scores))           
        scores_window.append(max_scores[-1])         
        moving_avgs.append(np.mean(scores_window))    # save moving average
        print('\rEpisode {} ({} sec)  -- \tMin: {:.3f}\tMax: {:.3f}\tMean: {:.3f}\tMov. Avg: {:.3f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]),end="") 
        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.3f}\tMax: {:.3f}\tMean: {:.3f}\tMov. Avg: {:.3f}'.format(\
                  i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))      
        if moving_avgs[-1] >= target_score*alter_factor and i_episode >= window_size:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.3f} over last {} episodes'.format(\
                                    i_episode, moving_avgs[-1], window_size))            
            if train_mode:
                stats+=f'solved{i_episode}'
                np.savetxt(stats+f'mean',mean_scores)
                np.savetxt(stats+f'min',min_scores)
                np.savetxt(stats+f'max',max_scores)
                np.savetxt(stats+f'avg',moving_avgs)
                for i,agent in enumerate(agent.maddpg_agent):
                    torch.save(agent.actor.state_dict(), stats+f'actor_ckpt_agent{i}.pth')
                    torch.save(agent.critic.state_dict(), stats+f'critic_ckpt_agent{i}.pth')
            break
            
    return mean_scores,min_scores,max_scores,moving_avgs,stats