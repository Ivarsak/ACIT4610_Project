#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy import integrate

env = gym.make('CartPole-v1')

def binary(x):
    if x >= 0:
        return 1
    else:
        return 0


state_log = []
pid_log = []
gen_mean = []
fit_log = []
best_gen = []


#--------------------------------------------
#calulate control Law + fitness/cost function
#--------------------------------------------

def fitness(Kp,Ki,Kd):
    
    theta_d = 0
    
    for i_episode in range(1):
        state = env.reset()
        integral = 0
        dt = 0
        e_last = 0
        delta_t = 0
        
        for t in range(500):
            env.render()
        
            e = state[2] - theta_d

            integral += e
            dt = e - e_last
            e_last = e

            pid =Kp * e + Ki * integral + Kd * dt
            action = binary(pid)
            state, reward, done, info = env.step(action)
            delta_t = t
            
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    return delta_t


#--------------------------------------------
             #Genetic Algorithm
#--------------------------------------------



def GA(Init, Population, Generation, Mutate_lower, Mutate_upper,Selection_size):

    Genomes = []
    gen_mean = []
    fit_log = []
    fit = []
    
  
    #Initialize population
    for s in range(Init):
        Genomes.append( (random.uniform(0,10),                       
                           random.uniform(0,0.01),
                           random.uniform(0,5)))
    
    #Run this loop for n-th generations
    for i in range(Generation):
    
        SelectPhenotype = []
        
        #Ranking soluations
        for s in Genomes:
            SelectPhenotype.append( (fitness(s[0],s[1],s[2]),s) )
            SelectPhenotype.sort()
            SelectPhenotype.reverse()
    
        
        print(f"=== Generation {i} best induvidual === ")
        print(SelectPhenotype[0])
        
        
        #Select best induviduals from the population
        selections = SelectPhenotype[:Selection_size]
    
        elements_s1 = []
        elements_s2 = []
        elements_s3 = []

        #Append the best solutions into arrays for each transition rule
        for s in SelectPhenotype:
            elements_s1.append(s[1][0])
            elements_s2.append(s[1][1])
            elements_s3.append(s[1][2])
        
        
        Genomes_new = []
        
        #Iterating induviduals in the population
        for _ in range(Population):
            Genome_s1 = random.choice(elements_s1) * random.uniform(Mutate_lower,Mutate_upper)
            Genome_s2 = random.choice(elements_s2) * random.uniform(Mutate_lower,Mutate_upper)
            Genome_s3 = random.choice(elements_s3) * random.uniform(Mutate_lower,Mutate_upper)
        
     
            Genomes_new.append((Genome_s1,Genome_s2,Genome_s3))

        Genomes = Genomes_new
          
        
#-----------------------        
 # Main loop #
#-----------------------


Init_pop = 100
Pop = 50
Gen = 100
Mut_lowerlimit = 0.98
Mut_upperlimit = 1.02
selectionsize = 5
last_i = 0



#if __name__ == '__Run_GA__':
    
GA(Init_pop, Pop, Gen, Mut_lowerlimit, Mut_upperlimit, selectionsize)
    
    


# In[9]:


plt.plot(gen_mean)
#plt.plot(gen_mean[0:200])


# 

# In[78]:


np.savetxt('gen300_PID_mean_rev0.csv', gen_mean, delimiter=',')
#print(bestsolutions)


# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

gen300 = pd.read_csv("gen300_PID_mean_rev0.csv")
gen200 = pd.read_csv("gen200_PID_mean_rev0.csv")
gen100 = pd.read_csv("gen100_PID_mean_rev0.csv")

fig = plt.figure()
plt.plot(gen300,label="Population = 300")
plt.plot(gen200,label="Population = 200")
plt.plot(gen100,label="Population = 100")
plt.legend(loc="lower right")
fig.suptitle(r'Population mean fitness $\frac{\sum f_{i}}{p_{n}}$ vs Generations', fontsize=17)
plt.xlabel('Generations', fontsize=15)
plt.ylabel('Mean [seconds]', fontsize=15)

plt.savefig('PID_results_cartpole.eps', format='eps')


# In[80]:


plt.savefig('PID_results_cartpole.eps', format='eps')


# In[4]:





# In[5]:





# In[82]:





# In[ ]:




