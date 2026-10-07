# This script simulates a system with three nuclei A, B, C where A decays to B, B decays to C, and C decays to A. 
# If exposed to a neutron flux, C can decay into A

import numpy
import numpy as np
from matplotlib import pyplot as plt 
import random

# Setting the figure size
plt.rcParams['figure.figsize'] = [9, 5]


# Function to determine if a transition occurs based on the transition probabiility and a random number

def has_transitioned(prob):
    r = random.random()

    return r < prob


# Function to take the current state of an atom and a set of transition rules
# Determines if transitions occur and returns the new state if so

def evolveOne(currentState, rules):

    for rule in rules:
    
        initial_state = rule[0]
        final_state = rule[1]
        prob = rule[2]
    
        if currentState == initial_state:
            transition_occurs = has_transitioned(prob)
        
            if transition_occurs:
                    return final_state
    
    return currentState


# Function to evolve the system of nuclei over a number of time steps

def evolve_system(NA, NB, NC, rules, n_steps):
    state = (['A'] * NA)+(['B'] * NB)+(['C'] * NC)
    
    A_count = np.empty(n_steps+1, dtype=int) #set amounts for each time step to be zero initially
    B_count = np.empty(n_steps+1, dtype=int)
    C_count = np.empty(n_steps+1, dtype=int)
    
    A_count[0] = NA #allocate first amounts of particles
    B_count[0] = NB
    C_count[0] = NC
    
    for t in range(1, n_steps+1):
        
        new_state = [] #blank future state
        
        for current_state in state: #cycle through the particles
            
            next_state = current_state #if no transition the state stays the same
        
            for (initial_state, final_state, prob) in rules:
                if current_state == initial_state and has_transitioned(prob):
                    next_state = final_state #same method as earlier
                    break
            
            new_state.append(next_state) #add updated state for the timestep
            
        state = new_state #replace the state of the timestep with the updated state
        
        A_count[t] = state.count('A') #for each timestep, record the number of each particle
        B_count[t] = state.count('B')
        C_count[t] = state.count('C')
    
    return A_count, B_count, C_count


# Plotting

# Plot of the number of nuclei over time for a system with and without neutron flux


n_steps = 200
t_total = 100
dt = t_total/n_steps

t_half_A = 10.1
t_half_B = 15.7
t_half_C = 3.2

NA=0
NB=0
NC=250


def probability_decay(halflife,t):
    return 1 - np.exp(-t*np.log(2)/ halflife)

prob_a_to_b = probability_decay(t_half_A, dt)
prob_b_to_c = probability_decay(t_half_B, dt)
prob_c_to_a = probability_decay(t_half_C, dt)

A_count = [NA]
B_count = [NB]
C_count = [NC]

rules_with_flux = [ 
    ('A', 'B', prob_a_to_b),
    ('B', 'C', prob_b_to_c),
    ('C', 'A', prob_c_to_a)]

A_count, B_count, C_count = evolve_system(NA, NB, NC, rules_with_flux, n_steps)

rules_without_flux = [ 
    ('A', 'B', prob_a_to_b),
    ('B', 'C', prob_b_to_c)]

NA_phase2 = A_count[-1]
NB_phase2 = B_count[-1]
NC_phase2 = C_count[-1]

A_count_phase2, B_count_phase2, C_count_phase2 = evolve_system(NA_phase2, NB_phase2, NC_phase2, rules_without_flux, n_steps)

phase1times = np.linspace(0,t_total, n_steps+1)
phase2times = np.linspace(t_total,2*t_total, n_steps+1)
complete_times = np.concatenate((phase1times, phase2times[1:]))

totalA_count = np.concatenate((A_count, A_count_phase2[1:]))
totalB_count = np.concatenate((B_count, B_count_phase2[1:]))
totalC_count = np.concatenate((C_count, C_count_phase2[1:]))



plt.figure(figsize=(12,6))
plt.plot(complete_times,totalA_count, label= "A nuclei")
plt.plot(complete_times,totalB_count, label= "B nuclei")
plt.plot(complete_times,totalC_count, label= "C nuclei")
plt.axvspan(0,100, color = 'red', alpha =0.1, label = 'With Neutron Flux')
plt.axvspan(100,200, color = 'blue', alpha =0.1, label = 'Without Neutron Flux')
plt.xlabel("Time (hours)")
plt.ylabel("Number of Nuclei")
plt.title("Decay of Nuclei With and Without Neutron Flux")
plt.legend()
plt.show()


# Simulating the system multiple times to calculate an average and uncertainty in the number of A atoms as a function of time

nsim = 20


total_A = []

for i in range(nsim):
    A_count, B_count, C_count = evolve_system(NA, NB, NC, rules_with_flux, n_steps)
    NA_phase2 = A_count[-1]
    NB_phase2 = B_count[-1]
    NC_phase2 = C_count[-1]
    A_count_phase2, B_count_phase2, C_count_phase2 = evolve_system(NA_phase2, NB_phase2, NC_phase2, rules_without_flux, n_steps)
    
    combined_As = np.concatenate((A_count, A_count_phase2[1:]))
    total_A.append(combined_As)

A_average = np.average(total_A, axis = 0)
A_std = np.std(total_A, axis = 0)

total_time = np.linspace(0, 2*t_total, 2*n_steps+1)


plt.figure(figsize=(12,6))
plt.errorbar(total_time, A_average, yerr=A_std, label= "A nuclei")
plt.axvspan(0,100, color = 'red', alpha =0.1, label = 'With Neutron Flux')
plt.axvspan(100,200, color = 'blue', alpha =0.1, label = 'Without Neutron Flux')
plt.xlabel("Time (hours)")
plt.ylabel("Number of Nuclei")
plt.title("Average Number of \"A\" Nuclei over 20 Simulations")
plt.legend()
plt.show()