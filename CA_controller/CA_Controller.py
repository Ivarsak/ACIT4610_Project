#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import gym
import matplotlib.pyplot as plt
#from Elementary_CA import rule_index, CA_run
from time import sleep
import scipy.integrate as integrate
from scipy import integrate

CA_steps = 2
fitness = 0
fitness_threshold = 500


env = gym.make('CartPole-v1')


#----------------
#Cellular Automata
#----------------
    
#Functions for computing elemetary cellular automata
#The development of the CA functions below originates from:
#https://matplotlib.org/matplotblog/posts/elementary-cellular-automata/


def rule_index(triplet):
    L, C, R = triplet
    index = 7 - (4*L + 2*C + R)
    return int(index)

#CA_run() computes a 1D CA grid
#Inputs: - Observerd states from the openai gym enviroment, these are encoded to binary values
#        - Number of steps for the CA to compute
#        - A vector containing elementary transition rules for each cell, which is evolved by a genetic algorithm

def CA_run(initial_state, n_steps, rule_grid):

    m_cells = len(initial_state)
    CA_run = np.zeros((n_steps, m_cells))
    CA_run[0, :] = initial_state
    
    grid = rule_grid

    for step in range(1, n_steps):
        
        for dx in range(0,m_cells):
        
            rule_string = np.binary_repr(int(grid[step][dx]), 8)
            rule = np.array([int(bit) for bit in rule_string])
            all_triplets = np.stack(
                [
                    np.roll(CA_run[step - 1, :], 1),
                    CA_run[step - 1, :],
                    np.roll(CA_run[step - 1, :], -1),
                ]
            ) 
            CA_run[step, :] = rule[np.apply_along_axis(rule_index, 0, all_triplets)]
            
            index_max = np.argmax(CA_run[n_steps-1])
            force_max = CA_run[n_steps-1][index_max]
        
            index_min= np.argmin(CA_run[n_steps-1])
            force_min = CA_run[n_steps-1][index_min]
        
            if force_min != force_max:
                if random.random() >= 0.5:
                    force_out = 1
                else:
                    force_out = 0
                
            else:
                force_out = force_max
            
    return force_out


#Encoding observations into binary values

def binary(x):
    if x >= 0:
        return 1
    else:
        return 0

# Running the openAI gym 
# Inputs: CA rules (s1,s2,s3,s4)
# Outputs: fitness

def fitness(s1,s2,s3,s4):
    
    for steps in range(1):
        state = env.reset()
        delta_t = 0
        S_rules = np.array([[0,0,0,0],[s1,s2,s3,s4]])
        
        for t in range(fitness_threshold):
            
            #env.render()
            initialize = np.array([binary(state[2]),binary(state[1]),binary(state[3]),binary(state[0])])
            action = int(CA_run(initialize,CA_steps,S_rules))
            state, reward, done, info = env.step(action)
            delta_t += reward 
            
            if done:
                print("Induvidual fitness",delta_t)
                break
                env.close()  

    return delta_t

    
#--------------------------------------------
             #Genetic Algorithm
#--------------------------------------------

#The code presented in https://www.youtube.com/watch?v=4XZoVQOt-0I was used as inspiration for the GA below

def GA(Init, Max_rule, Min_rule, Population, Generation, Mutate_lower, Mutate_upper,Selection_size):

    Genomes = []
    gen_mean = []
    fit_log = []
    fit = []
    
  
    #Initialize population
    for s in range(Init):
        Genomes.append( (random.randint(Min_rule, Max_rule),                       
                           random.randint(Min_rule, Max_rule),
                           random.randint(Min_rule, Max_rule),
                           random.randint(Min_rule, Max_rule)))
    
    #Run this loop for n-th generations
    for i in range(Generation):
    
        SelectPhenotype = []
        
        #Ranking soluations
        for s in Genomes:
            SelectPhenotype.append( (fitness(s[0],s[1],s[2],s[3]),s) )
            SelectPhenotype.sort()
            SelectPhenotype.reverse()
    
        
        print(f"=== Generation {i} best induvidual === ")
        print(SelectPhenotype[0])
        
        
        #Select best induviduals from the population
        selections = SelectPhenotype[:Selection_size]
    
        elements_s1 = []
        elements_s2 = []
        elements_s3 = []
        elements_s4 = []

        #Append the best solutions into arrays for each transition rule
        for s in SelectPhenotype:
            elements_s1.append(s[1][0])
            elements_s2.append(s[1][1])
            elements_s3.append(s[1][2])
            elements_s4.append(s[1][3])
        
        
        newGen = []
        
        #Iterating induviduals in the population
        for _ in range(Population):
            Genome_s1 = random.choice(elements_s1) * random.uniform(Mutate_lower,Mutate_upper)
            Genome_s2 = random.choice(elements_s2) * random.uniform(Mutate_lower,Mutate_upper)
            Genome_s3 = random.choice(elements_s3) * random.uniform(Mutate_lower,Mutate_upper)
            Genome_s4 = random.choice(elements_s4) * random.uniform(Mutate_lower,Mutate_upper)
        
     
            newGen.append((Genome_s1,Genome_s2,Genome_s3,Genome_s4))

        Genomes = newGen
         
        
        
#-----------------------        
 # Main loop #
#-----------------------


Init_pop = 50
rule_lowerlimit = 1
rule_upperlimit = 150
Pop = 100
Gen = 100
Mut_lowerlimit = 0.98
Mut_upperlimit = 1.02
selectionsize = 10
last_i = 0


#if __name__ == '__CA_Controller__':
    
GA(Init_pop, rule_upperlimit,rule_lowerlimit, Pop, Gen, Mut_lowerlimit, Mut_upperlimit, selectionsize)
    
    






