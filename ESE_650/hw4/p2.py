import gym
import numpy as np
import torch as th
import torch.nn as nn
import copy 
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import deque

def rollout(e, q, eps=0, T=200, render=False):
    traj = []

    x = e.reset()
    for t in range(T):
        if render:
            e.render()
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        u = u.int().numpy().squeeze()

        xp,r,d,info = e.step(u)
        t = dict(x=x,xp=xp,r=r,u=u,d=d,info=info)
        x = xp
        traj.append(t)
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )
    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0):
        # 1. get q values for all controls
        q = s.m(x)

        ### TODO: XXXXXXXXXXXX
        # eps-greedy strategy to choose control input
        # note that for eps=0
        # you should return the correct control u
        
        if (np.random.uniform(0, 1) <= 1-eps):
            u = th.argmax(q)
            # print(q)
            # print(u)
        else:
            actions = th.Tensor([0,1])
            u = actions[np.random.randint(0, s.udim)]
        return u

def loss(q, qc, ds):
    ### TODO: XXXXXXXXXXXX
    # 1. sample mini-batch from dataset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f
    batch_size = 1500
    gamma = .99
    
    random_trajectories = np.random.randint(0, len(ds), batch_size)
    # print(random_trajectories[0])
    random_steps = np.array([ds[i][np.random.randint(0, len(ds[i]))] for i in random_trajectories])
    # print(random_steps[0])
    terminals = th.FloatTensor([step['d'] for step in random_steps]).view(-1, 1)
    # print(terminals[0])
    states = th.FloatTensor([step['x'] for step in random_steps])
    # print(states[0])
    actions = th.LongTensor(np.array([step['u'] for step in random_steps]))
    # print(actions[0])
    next_states = th.FloatTensor([step['xp'] for step in random_steps])
    # print(next_states[0])
    rewards = th.FloatTensor([step['r'] for step in random_steps]).view(-1, 1)
    # print(rewards[0])
    
    u_primes = th.argmax(q(next_states), dim=1).detach()
    # print(q(next_states[0]))
    # print(actions[0:10])
    # print(u_primes[0])
    q_current = q(states).gather(1, actions.unsqueeze(-1))
    # print(q(states)[0])
    # print(q_current[0])
    q_next = qc(next_states).detach().gather(1, u_primes.unsqueeze(-1))
    # print(q_next[0])
    # print(qc(next_states[0]))
        
        # a = (q(state)[action]  - reward - gamma*(1-terminal)*th.max(q(next_state)))
    # print(((q_current  - (rewards + gamma*(1-terminals)*q_next))**2)[0])
    q_target = (rewards + gamma*(1-terminals)*q_next)
    # print(q_next[0:10])
    
    f = th.mean((q_current  - q_target)**2) 
    # print(q_target[0:10])
    # print(q_current[0:10])
        
    # f = f/batch_size
    
    return f

def evaluate(q):
    ### TODO: XXXXXXXXXXXX
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment to take control actions. Remember that
    # you should not perform epsilon-greedy exploration in the evaluation
    # phase
    # and report the average discounted
    # return of these 100 trajectories
    e = gym.make('CartPole-v1')
    iterations = 100
    all_rs = []
    
    for i in range(0, iterations):
        traj = rollout(e, q, eps=0, T=200)
        total_r = len(traj)
        all_rs.append(total_r)
    
    r = np.mean(all_rs)
    
    return r

if __name__=='__main__':
    e = gym.make('CartPole-v1')

    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n

    q = q_t(xdim, udim, 8)
    qc = copy.deepcopy(q)
    
    # Adam is a variant of SGD and essentially works in the
    # same way
    optim = th.optim.Adam(q.parameters(), lr=5e-3,
                          weight_decay=1e-4)

    ds = []
    val_avg_returns = []
    train_avg_returns = []
    losses = []

    # collect few random trajectories with
    # eps=1
    for i in range(2000):
        ds.append(rollout(e, q, eps=1, T=200))
        
    past_10 = deque([0]*10)
    eps_decay = .1
    
    for i in range(10000):
        q.train()
        
        if i >= 1000:
            eps_decay = 0
        
        t = rollout(e, q, eps=eps_decay)
        ds.append(t)


        # perform weights updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer
        q.zero_grad()
        f = loss(q, qc, ds)
        losses.append(f.item())
        
        # for param in qc.parameters():
        #     print(param)
        f.backward()
        optim.step()
        # for param in qc.parameters():
        #     print(param)
        
        q_params = [param.data for param in q.parameters()]
        with th.no_grad():
            for k, param in enumerate(qc.parameters()):
                param.copy_((.95*param) + (.05*q_params[k]))
                # print(param)
                # param.copy_(param*np.random.randint(5,10))
            
        # for param in q.parameters():
        #     print(param)
            
        if i % 100 == 0 :
            print('Logging data: iteration {}'.format(i))
            print('Val Avg Return: {}'.format(np.mean(past_10)))
        train_total_return = 0
        for j in range(0, 10):
            t_train_test = rollout(e,q)
            train_total_return += len(t_train_test)
        train_avg_return = train_total_return/10
        train_avg_returns.append(train_avg_return)
        val_avg_return = evaluate(q)
        val_avg_returns.append(val_avg_return)  
        
        past_10.popleft()
        past_10.append(val_avg_return)
        if np.mean(past_10) >= 199:
            break


    print(train_avg_returns[0:20])
    print(train_avg_returns[i-20:i])
    print(val_avg_returns[0:20])
    print(val_avg_returns[i-20:i])
    
    plt.figure()
    plt.plot(list(range(1, i+2)), train_avg_returns)
    plt.title('Train Avg Returns Per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Return')
    plt.show()
    
    plt.figure()
    plt.plot(list(range(1, i+2)), val_avg_returns)
    plt.title('Validation Avg Returns Per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Return')
    plt.show()
    
    plt.figure()
    plt.plot(list(range(1, i+2)), losses)
    plt.title('Loss Per Iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    
    th.save(q.state_dict(), 'q_10k_iters_try_three.pt')
    
    # rollout(e, q, eps=0, T=200, render=True)
    # e.close()
        
    
