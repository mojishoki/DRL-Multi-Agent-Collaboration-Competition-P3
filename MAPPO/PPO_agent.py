import torch
import torch.optim as optim
import numpy as np
from model import *
class PPO_agent():
    def __init__(self,state_size,full_state_size,hidden_in,hidden_out,action_size,seed,lr_actor,lr_critic):
        self.actor=ActorPPO(state_size, action_size, seed, hidden_in,hidden_out).to(device)
        self.critic=CriticPPO(full_state_size, seed, hidden_in,hidden_out).to(device)
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
    def act(self, states, add_noise=True): #we have NOT implemented any noise here
        self.actor.eval() #to act means to evaluate, so turning off batchnorm and dropout application
        with torch.no_grad():
            output_act=self.actor(states[np.newaxis,:])[:-1]
        a_o=[output_act[i].squeeze(dim=0).detach().cpu().numpy() for i in range(len(output_act))]
        return a_o
    def optimizer_zero_grad(self):
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
    def optimizer_step(self):
        self.optimizer_actor.step()
        self.optimizer_critic.step()
    def crit(self,states): #1*48
        self.critic.eval() 
        with torch.no_grad():
            output_crit=self.critic(states)
        return output_crit.detach().cpu().numpy() #1*1
    def traj_to_probs(self,states_ep,actions_ep,agent_number): #receives one trajectory of inputs
        """ Taking in trajectories and applying policy on it """
        self.actor.train()
        self.critic.train()
        #need to reshape as shape os states is max_t*num_agents*sth 
        #actions is already of shape max_t * 2, states for actor input needs to be reshaped
        # for states it needs to be merged for both agents so it's max_t=batch
        states_crit_input = states_ep.reshape(states_ep.shape[0],-1)  # b*48
        states_act_input = states_ep.transpose((1,0,2))[agent_number] #b*24
        output_act=self.actor(states_act_input,actions_ep)
        output_crit=self.critic(states_crit_input)
        return (output_act,output_crit) #shape max_t*(2,1,2,1)
    