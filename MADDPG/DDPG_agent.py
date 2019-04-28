import torch
import torch.optim as optim
from noise_utils import *
from update_utils import *
from model import *
from configs import device

class DDPG_agent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, merge, full_state_size, full_action_size,
                 hidden_in_critic, hidden_out_critic1, hidden_out_critic2, seed, lr_actor, lr_critic, gauss):
        self.out_actor = out_actor
        self.actor = Actor(in_actor, out_actor, seed, hidden_in_actor, hidden_out_actor).to(device)
        self.critic = Critic(merge, full_state_size, full_action_size, seed, hidden_in_critic, hidden_out_critic1, hidden_out_critic2).to(device)
        self.target_actor = Actor(in_actor, out_actor, seed, hidden_in_actor, hidden_out_actor).to(device)
        self.target_critic = Critic(merge, full_state_size, full_action_size, seed, hidden_in_critic, hidden_out_critic1, hidden_out_critic2).to(device)
        self.gauss = gauss
        if self.gauss:
            self.noise = GaussianExplorationNoise()
        else:
            self.noise = OUNoise(out_actor, scale=1.)
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs, noise_step, noise=2.): 
        """ `noise`: the scale of the OUNoise """
        obs = torch.from_numpy(obs).to(device).unsqueeze(dim=0).float()
        self.actor.eval()
        with torch.no_grad():
            if self.gauss:
                action = self.actor(obs) + torch.tensor(0.5 * np.random.randn(1, self.out_actor)*self.noise.noise(noise_step )).float()
            else:
                action = self.actor(obs) + noise*self.noise.noise()
        self.actor.train()
        return torch.clamp(action,-1,1)

    def target_act(self, obs):
        """ action by the target actor """
        obs = obs.to(device)
        with torch.no_grad():
            action = self.target_actor(obs) 
        return action