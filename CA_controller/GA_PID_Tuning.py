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
# The PID controller below originates from the code in: https://gist.github.com/HenryJia/23db12d61546054aa43f8dc587d9dc2c

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
#The code presented in https://www.youtube.com/watch?v=4XZoVQOt-0I was used as inspiration for the GA presented below


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
    
    
