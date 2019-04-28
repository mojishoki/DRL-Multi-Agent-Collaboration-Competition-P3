import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DDPG_agent import *
from ReplayBuffer import *
from configs import *

class MADDPG_agent:
    def __init__(self,env,brain_name,buffer_size = BUFFER_SIZE,batch_size = BATCH_SIZE,update_every = UPDATE_EVERY,
                 learn_num = LEARN_NUM,state_size = STATE_SIZE,
                 hidden_in_actor = HIDDEN_IN_ACTOR,hidden_out_actor = HIDDEN_OUT_ACTOR,out_actor = OUT_ACTOR,
                 merge = MERGE, full_state_size = FULL_STATE_SIZE, full_action_size = FULL_ACTION_SIZE,
                 hidden_in_critic = HIDDEN_IN_CRITIC,hidden_out_critic1 = HIDDEN_OUT_CRITIC1,hidden_out_critic2 = HIDDEN_OUT_CRITIC2,
                 gauss = GAUSS, seed = SEED,lr_actor=LR_ACTOR,lr_critic=LR_CRITIC,discount_factor=DISCOUNT_FACTOR, tau=TAU):
        """ Params
        ===============
            merge (bool)     : to merge the states and actions before feeding them to the critic
            gauss (bool)     : to use the gaussian noise for exploration instead of the OU noise
        """   
        self.env_info = env.reset(train_mode=True)[brain_name]
        self.num_agents=len(self.env_info.agents)
        self.hidden_in_actor=hidden_in_actor
        self.hidden_out_actor=hidden_out_actor
        self.hidden_in_critic=hidden_in_critic
        self.hidden_out_critic1=hidden_out_critic1
        self.hidden_out_critic2 = hidden_out_critic2
        self.merge=merge
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.maddpg_agent = [DDPG_agent(state_size,self.hidden_in_actor,self.hidden_out_actor,out_actor,merge,
                                       full_state_size, full_action_size,
                                       self.hidden_in_critic,self.hidden_out_critic1, self.hidden_out_critic2,seed,
                                       self.lr_actor,self.lr_critic,gauss) for i in range(self.num_agents)] 
                             
        self.seed=seed
        self.discount_factor = discount_factor
        self.tau = tau
        self.t_step = 0
        self.update_every=update_every
        self.learn_num=learn_num
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.memory=ReplayBuffer(out_actor, self.buffer_size, self.batch_size,self.seed)
        self.noise_step = 0
    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=1.):
        """get actions from all agents in the MADDPG object"""
        self.t_step = (self.t_step + 1) % self.update_every
        self.noise_step += 1
        actions = [agent.act(obs, self.noise_step, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return torch.stack(actions).squeeze().permute([1,0]).cpu().numpy()

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return torch.stack(target_actions).squeeze(dim=0)

    def step(self, obs, action, reward, next_obs, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(obs, action, reward, next_obs, done)
        # Learn, if enough samples are available in memory
        if self.t_step == 0 and self.noise_step>5500: #only after 
            if len(self.memory) > self.batch_size:
                #loop here to update many times for a update_every time_step
                for _ in range(self.learn_num):
                    samples = self.memory.sample()
                    for i in range(self.num_agents):
                        self.update(samples,i)
    
    def reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset()
        
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        obs,action,reward,next_obs,done=samples        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        target_actions = self.target_act(next_obs.permute([1,0,2]))
        #target_actions shape is 2(agent)*B*2(actions) 
        target_actions = target_actions.permute([1,0,2])
        with torch.no_grad():
            q_next=agent.target_critic(next_obs,target_actions)
        y = reward.permute([1,0])[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done.permute([1,0])[agent_number].view(-1, 1))
        q = agent.critic(obs, action)
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.) #uncomment this to allow clipping
        agent.critic_optimizer.step()
        agent.actor_optimizer.zero_grad()
        ####-------actor_update---------####
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs.permute([1,0,2])) ]
        q_input=torch.stack(q_input).squeeze(dim=0).permute([1,0,2])
        actor_loss = -agent.critic(obs, q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)