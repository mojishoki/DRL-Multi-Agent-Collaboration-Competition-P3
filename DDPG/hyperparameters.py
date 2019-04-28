device = "cuda"
# device= 'cpu'
#FOR DDPG
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # Weight_decay for Adam optimizer
UPDATE_EVERY = 5       # every update_every steps update
LEARN_NUM = 5          # learn_num times
SEED=7                 # random seed