import torch
import numpy as np

#taken froom https://github.com/ulamaca/DRLND_P3_MultiAgent_RL/blob/master/maddpg_agent.py
class GaussianExplorationNoise:
    '''
    Gaussian Exploration Noise for action space exploration:
        noise_scale is exponentially decaying as agents makes more steps (t), precisely
                       1.  e_0                               ,if t<t_0
        noise_scale =  2.  e_0 x alpha^( (t-t_0)//L_factor ) ,if t>t_0 and value of 2.<e_T
                       3.  e_T                               ,if value of 2.>e_T
    '''
    def __init__(self, epsilon_0 = 1, epsilon_end = 0.1, l_factor = 80, decay_factor = 0.999, t_0 = 5500):
        self.e_0=epsilon_0
        self.e_T=epsilon_end
        self.t_0=t_0
        self.alpha=decay_factor
        self.l_factor = l_factor
    def noise(self, t):
        if t<self.t_0:
            return self.e_0
        else:
            return max( self.e_0*self.alpha**( (t-self.t_0)//self.l_factor), self.e_T)
        
class OUNoise:
    def __init__(self, action_dimension, scale=1., mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(device)