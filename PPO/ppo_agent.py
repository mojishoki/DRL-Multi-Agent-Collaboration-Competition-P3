import torch
import torch.optim as optim
import numpy as np
from ppo_model import *


class AgentPPO():
    def __init__(self,state_size,action_size,seed=1,lr=3e-4):
        self.policy=ActorCriticPPO(state_size,action_size,seed).to(device)
        self.lr=lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
    def act(self, states, add_noise=True):
        self.policy.eval() #to act means to evaluate, so a change in batchnorm and dropout application
        with torch.no_grad():
            output=self.policy(states)[:-1]
        return [output[i].detach().cpu().numpy() for i in range(len(output))]
    def collect_trajectories(self,env, brain_name, max_t=1000, nrand = 10, n_traj=1):
        #only works for n_traj=1
        states_list,actions_list,old_log_probs_list,rewards_list,values_list = ([] for i in range(5))
        for i_episode in range(1, n_traj+1):
            env_info = env.reset(train_mode=True)[brain_name] 
            states = env_info.vector_observations
            for t in range(max_t):
                states_list.append(list())
                actions_list.append(list())
                rewards_list.append(list())
                old_log_probs_list.append(list())
                values_list.append(list())
                actions, old_log_probs, old_vs = self.act(states, add_noise=True)
                env_info = env.step(actions)[brain_name]    
                rewards = env_info.rewards    
                for state, action, old_log_prob, reward, old_v in zip(states, actions, old_log_probs, rewards,old_vs):
                    states_list[t].append(state)  
                    actions_list[t].append(action)
                    rewards_list[t].append([reward])
                    old_log_probs_list[t].append(old_log_prob)
                    values_list[t].append(old_v)
                states = env_info.vector_observations
                if np.any(env_info.local_done):   # exit loop when episode ends
                    break 
        return states_list , actions_list, old_log_probs_list, rewards_list, values_list #these are max_t*num_agents*(24,2,1,1,1) shape