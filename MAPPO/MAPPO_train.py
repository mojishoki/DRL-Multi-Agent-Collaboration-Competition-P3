import torch
import torch.nn
import numpy as np
import time
from model import device

def MAPPO_train(agent,env,brain_name,episode = 20000,max_t = 30000,
              SGD_epoch = 30, n_traj = 10, target_score = 0.5 , discount = .99, eta=0.95, epsilon = 0.2, c1 = 0.5, 
              beta = .01, delta=0.1, clip=0.2, loss_kind="entropy_exact",print_every=200):
    """
        epsiode: number of episodes to run
        max_t: max number of steps
        n_traj: number of trajectories to collect for SGD_epoch
        SGD_epoch: number of updates using the old policy trajectories
        target_score: score to reach to solve the env
        discount: discount for reward
        eta: the Generalized Advantage Estimation (GAE) hyperparameter usually between 0.92 to 0.98 (if 1., it is just MC)
        epsilon: the clipping ppo hyperparameter
        Loss scalar factors:
        c1: the scalar factor for value_loss default 0.5
        beta: the scalar factor for new policy entropy and also new policy distribution entropy
        delta: the scalar factor for new and old policy KL distance
        clip: clipping the gradients
        Loss type:
        loss_kind: can be either of ["simplest","entropy","entropy_exact","entropy_approximate","KL_approximate","KL_entropy_approximate"]
            simplest: No consideration of entropies or KL
            entropy: Consider both forms of entropies
            entropy_exact: Consider only new policy distribution entropy
            entropy_approximate: Consider only new policy entropy
            KL_approximate: Consider only new and old policy KL distance
            KL_entropy_approximate: merging entropy_approximate and KL_approximate
    """
    mean_rewards,min_rewards,max_rewards,mavg_rewards=([] for _ in range(4))
    stats=f'Checkpoints/'
    stats+=f'{discount}{eta}{epsilon}{beta}{c1}{delta}{max_t}{SGD_epoch}{loss_kind}{clip}{agent.lr_actor}{agent.lr_critic}'
    #preparing the discount and eta_discount arrays to compute future returns and GAE
    durations=[]
    for e in range(1,episode+1):
        start_time=time.time()
        states_list, actions_list,old_log_probs_list, rewards_list,v_list, done_list=agent.collect_trajectories(max_t=max_t,n_traj=n_traj)
        total_rewards = np.array([np.sum(rewards_ep,axis=0) for rewards_ep in rewards_list]) #per episode
        v_list=np.array(v_list)
        rewards_list=np.array(rewards_list)
        #Instead of computing these in surrogate as it is usually done, we do it here to save time
        returns_list,advantage_list = ([list() for _ in range(n_traj)] for _ in range(2)) #n_traj*sth*agent*1
        tmp_r = np.zeros((2,1))
        tmp_adv = np.zeros((2, 1))
        next_value=np.zeros((2,1))
        for m in range(n_traj):
            for i in reversed(range(len(states_list[m]))):
                if i<len(states_list[m])-1: 
                    next_value = v_list[m][i+1]
                tmp_r = rewards_list[m][i] + discount  * tmp_r
                tmp_td_error = rewards_list[m][i] + discount * next_value - v_list[m][i]
                tmp_adv = tmp_adv * eta * discount  + tmp_td_error
                returns_list[m].append(tmp_r)
                advantage_list[m].append(tmp_adv)
        #WE CANNOT DO BATCH NORM AS EPISODE LENGTHS ARE DIFFERENT
#         returns_list=np.array(returns_list)
#         advantage_list=np.array(advantage_list)
        #NORMALIZING:
#         mean = np.mean(returns, axis=1) 
#         std = np.std(returns, axis=1) + 1.0e-10
#         returns = (returns - mean[:,np.newaxis])/std[:,np.newaxis]
        for i in range(len(returns_list)):
            returns_list[i] = torch.tensor(np.array(returns_list[i]).transpose((1,0,2)), dtype=torch.float, device=device)
#             mean = np.mean(advantage_list, axis=1) 
#             std = np.std(advantage_ep, axis=1) + 1.0e-10
#             advantage_ep = (advantage_ep - mean[:,np.newaxis])/std[:,np.newaxis]
            advantage_list[i] = torch.tensor(np.array(advantage_list[i]).transpose((1,0,2)), dtype=torch.float, device=device)
            old_log_probs_list[i] = torch.tensor(old_log_probs_list[i].transpose((1,0,2)), dtype=torch.float, device=device)

        for _ in range(SGD_epoch):
            new_lists =  agent.traj_to_probs(states_list, actions_list)
            for agent_number in range(agent.num_agents):    
                L = -agent.surrogate(agent_number, new_lists, n_traj, old_log_probs_list, states_list,
                               actions_list,returns_list,advantage_list,
                                      c1=c1, epsilon=epsilon,delta=delta,
                                      beta1=beta,beta2=beta,loss_kind=loss_kind) #negative sign is important as maximizing is the goal
                agent.mappo_agents[agent_number].optimizer_zero_grad()
                L.backward()
                torch.nn.utils.clip_grad_norm_(agent.mappo_agents[agent_number].actor.parameters(),clip)
                torch.nn.utils.clip_grad_norm_(agent.mappo_agents[agent_number].critic.parameters(),clip) #clipping the gradients
                agent.mappo_agents[agent_number].optimizer_step()
                del L
        #decaying the clipping and scalar factor hyperparams 
        epsilon*=.9999
        beta*=.9995
        for i in range(n_traj):
            mean_rewards.append(np.mean(total_rewards,axis=1)[i])
            min_rewards.append(np.min(total_rewards,axis=1)[i])
            max_rewards.append(np.max(total_rewards,axis=1)[i])
            mavg_rewards.append(np.mean(max_rewards[-100:]))
        duration=time.time()-start_time
        durations.append(duration)
        print("\r Episode: {}, duration: {}, max:{}, min: {}, moving avg: {}".format(e,round(duration),np.max(total_rewards),
                                                                                     np.min(total_rewards),mavg_rewards[-1]),end="")
        if e%print_every==0:
            print("Episode: {}, durations: {}, max:{}, min: {}, moving avg: {}".format(e,round(sum(durations[-print_every:])),np.max(total_rewards),np.min(total_rewards),mavg_rewards[-1]))
        if mavg_rewards[-1]>target_score:
            print("Environment Solved in {} Epsiodes and {} time".format(e,round(sum(durations))))
            stats+=f'solved{e}'
            for i,agent in enumerate(agent.mappo_agents):
                torch.save(agent.actor.state_dict(),stats+f'actor{i}.pth')
                torch.save(agent.critic.state_dict(),stats+f'critic{i}.pth')
            np.savetxt(stats+f'mean', mean_rewards)
            np.savetxt(stats+f'min', min_rewards)
            np.savetxt(stats+f'max', max_rewards)
            np.savetxt(stats+f'mavg', mavg_rewards)
            break
    return mean_rewards,min_rewards,max_rewards,mavg_rewards,durations,stats