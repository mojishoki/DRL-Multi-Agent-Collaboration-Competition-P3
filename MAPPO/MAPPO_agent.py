import torch
import numpy as np
from PPO_agent import *


class MAPPO_agent():
    def __init__(self,env,brain_name,state_size,full_state_size,hidden_in,hidden_out,action_size,seed,lr_actor,lr_critic):
        env_info = env.reset(train_mode=True)[brain_name]
        self.env=env
        self.brain_name=brain_name
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.num_agents=len(env_info.agents)
        self.mappo_agents=[PPO_agent(state_size,full_state_size,hidden_in,
                                    hidden_out,action_size,seed,lr_actor,lr_critic) for _ in range(self.num_agents)]
        
    def act(self,states,add_noise=True):
        x=[agent.act(states[i][np.newaxis,:],add_noise=True) for i,agent in enumerate(self.mappo_agents)]
        return [np.array([x[i][j] for i in range(self.num_agents)]) for j in range(2)]
    
    def crit(self,states):
        x=[agent.crit(states.reshape(1,-1)) for agent in self.mappo_agents]
        return np.array([x[i].squeeze(axis=0) for i in range(len(x))]) # x[i] is a 1*1 thing and we want output to be 2*1
    
    def traj_to_probs(self,states_list,actions_list): #states_list shape M(list dim)*sth*2*24(sth*2*24 is numpy),
                                                        #actions_list M*sth*2(agents)*2
        x=[]
        for i,agent in enumerate(self.mappo_agents):
            y=[] #will contain all traj probs for one agent
            for states_ep,actions_ep in zip(states_list,actions_list):
                y.append(agent.traj_to_probs(states_ep,actions_ep.transpose((1,0,2))[i],i))
            x.append(y)
        return x #shape 2(agents)* M *(output_act,output_crit)
    
    def collect_trajectories(self, max_t=10000, nrand = 10, n_traj=1):
        states_list,actions_list,old_log_probs_list,rewards_list,values_list,done_list= ([] for i in range(6))
        for i_episode in range(1, n_traj+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name] 
            states = env_info.vector_observations
            states_ep,actions_ep,old_log_probs_ep,rewards_ep,values_ep = ([] for i in range(5))
            
            for t in range(max_t):
                actions, old_log_probs = self.act(states, add_noise=True) #2*1*2, 2*1*1
                old_vs = self.crit(states) #2*1 thing
                env_info = self.env.step(actions.squeeze(axis=1))[self.brain_name]    
                rewards = env_info.rewards    
                states_ep.append(states)  
                actions_ep.append(actions.squeeze(axis=1))
                rewards_ep.append(rewards)
                old_log_probs_ep.append(old_log_probs.squeeze(axis=1))
                values_ep.append(old_vs)
                states = env_info.vector_observations
                if np.any(env_info.local_done):   # exit loop when episode ends
                    dones_ep=np.zeros((len(env_info.agents),t))
                    dones_ep[:,-1]=np.array([1.,1.])
                    break
            states_list.append(np.array(states_ep))
            actions_list.append(np.array(actions_ep))
            rewards_list.append(np.array(rewards_ep))
            old_log_probs_list.append(np.array(old_log_probs_ep))
            values_list.append(np.array(values_ep))
            done_list.append(dones_ep)
        return states_list , actions_list, old_log_probs_list, rewards_list, values_list, done_list
    #these are n_traj(list len)*sth(max_t's)*2(num_agent)*(24,2,1,2,1,1) shape
    
    def surrogate(self, agent_number,new_lists,n_traj, old_log_probs_list,
                  states_list, actions_list,returns_list,advantage_list,epsilon=0.2, c1 = 0.5, beta1=0.01,
                     beta2=0.01,delta= 0.1, loss_kind="entropy"):
        """ 
        Surrogate function implementing the loss function.
        """
        actions_list, new_log_probs_list, new_entropy_list, new_v_list = ([] for _ in range(4))  
        Lsur_clipped=[]
        value_loss=[]
        new_entropy=[]
        new_policy_entropy=[]
        new_old_policy_KL=[]
        for m in range(n_traj): 
            output_act, new_v_ep = new_lists[agent_number][m]
            actions_ep, new_log_probs_ep, new_entropy_ep = output_act
            ratio_ep = (new_log_probs_ep-old_log_probs_list[m][agent_number]).exp()
            ratio_clamped_ep = torch.clamp(ratio_ep, 1-epsilon, 1+epsilon)
            Lsur_ep = advantage_list[m][agent_number]*ratio_ep
            Lsur_clamped_ep = advantage_list[m][agent_number]*ratio_clamped_ep
            
            Lsur_clipped.append(torch.min(Lsur_ep,Lsur_clamped_ep).mean())
            value_loss.append(torch.mean((new_v_ep-returns_list[m][agent_number])**2))
            new_entropy.append(torch.mean(new_entropy_ep))
            new_policy_entropy.append(torch.mean(new_log_probs_ep.exp()*new_log_probs_ep))
            new_old_policy_KL.append(torch.mean(new_log_probs_ep.exp()*(new_log_probs_ep-old_log_probs_list[m][agent_number])))
        Lsur_clipped=torch.stack(Lsur_clipped).squeeze(dim=0)
        value_loss=torch.stack(value_loss).squeeze(dim=0)
        new_entropy=torch.stack(new_entropy).squeeze(dim=0)
        new_policy_entropy=torch.stack(new_policy_entropy).squeeze(dim=0)
        new_old_policy_KL=torch.stack(new_old_policy_KL).squeeze(dim=0)
        if loss_kind == "simplest":
            return torch.mean(Lsur_clipped-c1*value_loss)
        if loss_kind == "entropy":
            return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy+beta2*new_entropy)
        if loss_kind == "KL_entropy_approximate":
            return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy-delta*new_old_policy_KL)
        if loss_kind == "entropy_exact":
            return torch.mean(Lsur_clipped-c1*value_loss+beta2*new_entropy)
        if loss_kind == "entropy_approximate":
            return torch.mean(Lsur_clipped-c1*value_loss+beta1*new_policy_entropy)
        if loss_kind == "KL_approximate":
            return torch.mean(Lsur_clipped-c1*value_loss-delta*new_old_policy_KL)