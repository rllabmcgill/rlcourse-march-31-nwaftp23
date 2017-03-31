import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from random import shuffle

# Create arms
# parameters for explore then commit
# K number of arms
# mu array of means
# s covariance matrix
# m number of times each arm is pulled before commit
# n total pulls
K = 10
np.random.seed(4)
m = 100
n = 10**4

tot_regret= np.zeros((6,n))
num = 100
for l in range(num):
    print('experiment ', l+1)
    mu = np.sort(np.random.uniform(0,1,K))
    lol = np.tile(mu,(n,1))
    data = np.random.rand(n,K) < lol
    tot = np.sum(data[0:(m+1),:] , axis=0)
    mu_hat = tot/m


    winner=np.argmax(mu_hat)

    regret_etc = np.zeros(n)
    for i in range(n):
        if i < m*K:
            regret_etc[i] = regret_etc[i-1] + np.max(mu) - mu[int(np.floor(i/m))]
        else:
            regret_etc[i] = regret_etc[i-1] + np.max(mu) - mu[winner]
    tot_regret[0,:] += regret_etc


    regret_ran = np.zeros(n)
    for i in range(n):
        regret_ran[i] = regret_ran[i-1] + np.max(mu) - mu[np.random.choice(K)]

    tot_regret[1,:] += regret_ran


    eps = 0.01
    regret_eps = np.zeros(n)
    mu_hat2 = np.zeros(K)
    count = np.zeros(K)
    for i in range(n):
        W = np.argwhere(mu_hat2 == np.amax(mu_hat2))
        ind = [i for i in list(range(K)) if i not in list(W[:,0])]
        if len(W) == K :
            rr = np.random.choice(K)
            regret_eps[i] = regret_eps[i-1] + np.max(mu) - mu[rr]
            count[rr] += 1
            rew = data[i,rr]
            mu_hat2[rr] += 1/count[rr]*(rew-mu_hat2[rr])
        else:
            a = np.random.uniform(0,1)
            if a >= eps :
                np.random.shuffle(W)
                regret_eps[i] = regret_eps[i-1] + np.max(mu) - mu[W[0][0]]
                count[W[0][0]] += 1
                rew = data[i,W[0][0]]
                mu_hat2[W[0][0]] = 1/count[W[0][0]]*(mu_hat2[W[0][0]]*(count[W[0][0]]-1)+rew)
            else:
                shuffle(ind)
                regret_eps[i] = regret_eps[i-1] + np.max(mu) - mu[ind[0]]
                count[ind[0]] += 1
                rew = data[i,ind[0]]
                mu_hat2[ind[0]] = mu_hat2[ind[0]]+1/count[ind[0]]*(rew-mu_hat2[ind[0]])

    tot_regret[2,:] += regret_eps

    regret_ucb = np.zeros(n)
    count2 = np.zeros(K)
    mu_hat3 = np.zeros(K)
    c = .125
    for i in range(n):
        if i < K :
            regret_ucb[i] = regret_ucb[i-1] + np.max(mu) - mu[i]
            count2[i] += 1
            rew = data[i,i]
            mu_hat3[i] = 1/count2[i]*(mu_hat3[i]*(count2[i]-1)+rew)
        else:
            At = mu_hat3 + ((c*np.log(i)/count2)**0.5)
            winner = np.argwhere(At == np.amax(At))
            np.random.shuffle(winner)
            w = winner[0][0]
            regret_ucb[i] = regret_ucb[i-1] + np.max(mu) - mu[w]
            count2[w] += 1
            rew = data[i,w]
            mu_hat3[w] = 1/count2[w]*(mu_hat3[w]*(count2[w]-1)+rew)


    tot_regret[3,:] += regret_ucb

    regret_Thomp = np.zeros(n)
    alpha = np.ones(K)
    beta = np.ones(K)
    check = 0
    for i in range(n):
        s_list=[]
        for j in range(K):
            s = sum(np.random.beta(alpha[j],beta[j],1))
            s_list.append(s)
        winn = np.argmax(s_list)
        regret_Thomp[i] = regret_Thomp[i-1] + np.max(mu) - mu[winn]
        x = data[i,winn]
        alpha[winn] += x
        beta[winn] += 1-x


    tot_regret[4,:] += regret_Thomp

    eps2 = 0
    regret_o_eps = np.zeros(n)
    mu_hat4 = np.ones(K)
    count3 = np.ones(K)
    for i in range(n):
        W = np.argwhere(mu_hat4 == np.amax(mu_hat4))
        ind = [i for i in list(range(K)) if i not in list(W[:,0])]
        if len(W) == K :
            rr = np.random.choice(K)
            regret_o_eps[i] = regret_o_eps[i-1] + np.max(mu) - mu[rr]
            count3[rr] += 1
            rew = data[i,rr]
            mu_hat4[rr] += 1/count3[rr]*(rew-mu_hat4[rr])
        else:
            a = np.random.uniform(0,1)
            if a >= eps2 :
                np.random.shuffle(W)
                regret_o_eps[i] = regret_o_eps[i-1] + np.max(mu) - mu[W[0][0]]
                count3[W[0][0]] += 1
                rew = data[i,W[0][0]]
                mu_hat4[W[0][0]] = 1/count3[W[0][0]]*(mu_hat4[W[0][0]]*(count3[W[0][0]]-1)+rew)
            else:
                shuffle(ind)
                regret_o_eps[i] = regret_o_eps[i-1] + np.max(mu) - mu[ind[0]]
                count3[ind[0]] += 1
                rew = data[i,ind[0]]
                mu_hat4[ind[0]] = mu_hat4[ind[0]]+1/count3[ind[0]]*(rew-mu_hat4[ind[0]])

    tot_regret[5,:] += regret_o_eps


ave=tot_regret/num
print(tot_regret[5,:])
print(tot_regret[1,:])
# upper_bound = m * np.sum(np.max(mu)-mu) + (n-m*K)*np.dot(np.max(mu)-mu,np.exp(-m*((np.max(mu)-mu)**2)/4))
#print(upper_bound)
#print(mu)
line1=plt.semilogy(ave[0,:], label='ETC')
# line2=plt.axhline(y=upper_bound,xmin=0,xmax=n, color='r',ls='--',label='upper_bound')
line3=plt.semilogy(ave[1,:], label='random')
line4=plt.semilogy(ave[2,:], label='Epsilon-Greedy')
line5=plt.semilogy(ave[3,:], label='UCB with c = .5')
line6=plt.semilogy(ave[4,:], label='Thompson Sampling')
line7=plt.semilogy(ave[5,:], label='Optimistic Epsilon-Greedy')
plt.title('Simulated Bandit Performance for K = 10')
plt.ylabel('Expected Total Regret')
plt.xlabel('Round Index')
plt.legend()
plt.show()
