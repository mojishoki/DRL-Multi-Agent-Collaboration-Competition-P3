device = 'cpu' #default is cpu, comment this and uncomment below if you want to change it
# device = 'cuda'
#MADDPG configs
BUFFER_SIZE=int(5e4)
BATCH_SIZE=200
UPDATE_EVERY=2
LEARN_NUM=4
STATE_SIZE=24
ACTION_SIZE=2
HIDDEN_IN_ACTOR=256
HIDDEN_OUT_ACTOR=256
OUT_ACTOR=2
MERGE=False
FULL_STATE_SIZE=24*2
FULL_ACTION_SIZE=2*2
HIDDEN_IN_CRITIC=256
HIDDEN_OUT_CRITIC1=256
HIDDEN_OUT_CRITIC2 = 128
GAUSS=True
SEED=70
LR_ACTOR=0.0001
LR_CRITIC=0.0005
DISCOUNT_FACTOR=0.99
TAU=0.001

#MADDPG_train configs
RATIO=.5
HARD_END=300
NOISE=5.
NOISE_END=0.1
ALTER_FACTOR=1.
