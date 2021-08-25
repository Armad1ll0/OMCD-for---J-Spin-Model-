# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:16:57 2021

@author: amill
"""

#Note: this algorithm is basic at the moment as I am making the fundamental consituent parts but will become more complex over time 
#need to turn all of these into function if I can and expand it to 3d lattices if possible 
#at the moment its basically a MCM algorithm 
import time 
start = time.time()
import numpy as np 
import random as random 
import matplotlib.pyplot as plt
import math as math 
from collections import Counter 
import multiprocessing
from collections import defaultdict
import collections 
import os 

#size of 3d lattice 
nx = 3

#quick graph to show the map of the up and down particles, 1=red and 0=blue. 
def initial_config(spin_glass_box_original):
    x, y, z = (spin_glass_box_original>-1).nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, x in np.ndenumerate(spin_glass_box_original):
        if x == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, y in np.ndenumerate(spin_glass_box_original):
        if y == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, z in np.ndenumerate(spin_glass_box_original):
        if z == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, x in np.ndenumerate(spin_glass_box_original):
        if x == -1:
            ax.scatter(*index, s=5, c = 'blue')
    for index, y in np.ndenumerate(spin_glass_box_original):
        if y == -1:
            ax.scatter(*index, s=5, c = 'blue')
    for index, z in np.ndenumerate(spin_glass_box_original):
        if z == -1:
            ax.scatter(*index, s=5, c = 'blue')
    plt.title('Initial Model for 3D +/- J Model for OMCD')
    return plt.show()

#deflation number 
d0 = (nx*nx*nx)-1
#d0 = math.ceil(((nx*nx*nx)/50))
d0 = math.ceil((nx**3)/2)
#number of rejections before we lower d0
num_rejections_lower_d0 = 800
#this needs to be 
monte_carlo_step = ((nx**3)/d0)
number_steps = 1000

monte_carlo_steps = round(number_steps*monte_carlo_step)

#NOTE: These constants need to be changed based off scientific observation. We are not there though yet. 

#energy equation 
#H = -J*S_i*S_j, where J = +/-1 randomly 
J_values = [1, -1]

#this function to return the adjacent values of the point xyz
horiz_vert= lambda x,y,z:[((x+1)%nx,y,z), (x,(y+1)%nx,z), (x,y,(z+1)%nx)]

#energy calculator, I think this works faster and is correct, old document in file name: omcd lattice (wrong maybe) 
#has the old code and I think is wrong. 
#it is also noteworthy that the old calculation was a lot slower than this lambda function 
def calc_total_energy(zero_matrix, J_matrix):
    total_energy = 0        
    for x in range(len(zero_matrix)):
        for y in range(len(zero_matrix)):
            for z in range(len(zero_matrix)):
                neighbours = horiz_vert(x, y, z)
                for i in neighbours:
                    #I do this as I know from the horiz_vert function, the first entry of neighbours gives the +x, 2nd, +y, 
                    #and 3rd, +z so they correspond to the Jij link matrices below. 
                    #this is the cost function of H = -J*S_i*S_j, where J = +/-1 randomly 
                    if neighbours.index(i)==0:
                        total_energy += -1*(J_matrix[x, y, z][0])*zero_matrix[x, y, z]*zero_matrix[i]
                    elif neighbours.index(i)==1:
                        total_energy += -1*(J_matrix[x, y, z][1])*zero_matrix[x, y, z]*zero_matrix[i]
                    elif neighbours.index(i)==2: 
                        total_energy += -1*(J_matrix[x, y, z][2])*zero_matrix[x, y, z]*zero_matrix[i]
                     
                    #print(J_matrix[x, y, z])
                    #total_energy += -1*J_matrix[x, y, z]*zero_matrix[x, y, z]*zero_matrix[i]
    #print('The total_energy is ' + str(total_energy))
    #print(total_energy)
    return(total_energy)              

#NOTE: Jij are random but fixed for the configurations 

def OMCD(spin_glass_box_original, spin_glass_box_medium, J_matrix, initial_total_energy, medium_total_energy, graph_energy, d0, d0_start_value, acceptance, rejection, monte_carlo_steps):
    energy_change = []
    d0_when_energy_change = []
    count = 0
    while d0 > 0:
        #creating a copy of the spin_glass_medium to flip 
        spin_glass_box_newest = spin_glass_box_medium.copy()
        #print(spin_glass_box_newest)

        flip_coords = []
        for x in range(0, nx):
            for y in range(0, nx):
                for z in range(0, nx):
                    flip_coords.append([x, y, z])
        
        random.shuffle(flip_coords)

        flip_these = flip_coords[0:d0]
        #print(d0)
        #print('We are flipping these coordinates')
        #print(flip_these)
        #print(spin_glass_box_newest)
        for j in flip_these:
            a = j[0]
            b = j[1]
            c = j[2]
            spin_glass_box_newest[a][b][c] = -1*spin_glass_box_newest[a][b][c]
        
        #calculating the new energy of the flipped spin glasses 
        newest_total_energy = calc_total_energy(spin_glass_box_newest, J_matrix)
      
        #functions says whether to accept (if lower) or reject (if higher) new system energy 
        if newest_total_energy <= medium_total_energy:
            energy_change.append(newest_total_energy - medium_total_energy)
            d0_when_energy_change.append(d0)
            medium_total_energy = newest_total_energy
            spin_glass_box_medium = spin_glass_box_newest
            acceptance += 1
            count += 1
            graph_energy.append(medium_total_energy)
            #print('This energy has been Accepted! Woooooooo!')
            #print('The new energy is ' + str(medium_total_energy))
        else: 
            #print('Gonna have to reject this one, sorry bud.')
            rejection += 1
            count += 1
        
        #this if statement changes the number of spins we want to change but at the moment it shortens the length of the new spin system instead
        #read documentation on random.sample() to finish this part 
        if count == monte_carlo_steps:
            d0 = d0-1 
            #print('do within this Schedule is now ' + str(d0))
            #print('To many rejections in a row, lets try lowering the number to see if that works. The new value of d0 is ' + str(d0))
            count = 0
            if d0 != 0:
                monte_carlo_step = ((nx**3)/d0)
                number_steps = 1000
                monte_carlo_steps = round(number_steps*monte_carlo_step)
                #print(monte_carlo_steps)
            elif d0 == 0:
                #print('d0 will not go any lower mate, gonna have to stop here.')
                break 
    
    #gives acceptance and rejection stats for this particular configuration 
    #print('The number of acceptances was ' + str(acceptance) + ' and the number of rejections was ' + str(rejection))
    #print('so our ratio of accepted to total number of attempted moves is ' + str(acceptance*100/(acceptance + rejection)) + '%')
    print('Sample Complete')
    return spin_glass_box_medium, medium_total_energy, energy_change, d0_when_energy_change


#quick graph to show the map of the up and down particles after all the changes in configuration, 1=green and 0=blue. 
def final_config(spin_glass_box_medium):
    x, y, z = (spin_glass_box_medium>-1).nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, x in np.ndenumerate(spin_glass_box_medium):
        if x == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, y in np.ndenumerate(spin_glass_box_medium):
        if y == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, z in np.ndenumerate(spin_glass_box_medium):
        if z == 1:
            ax.scatter(*index, s=5, c = 'red')
    for index, x in np.ndenumerate(spin_glass_box_medium):
        if x == -1:
            ax.scatter(*index, s=5, c = 'blue')
    for index, y in np.ndenumerate(spin_glass_box_medium):
        if y == -1:
            ax.scatter(*index, s=5, c = 'blue')
    for index, z in np.ndenumerate(spin_glass_box_medium):
        if z == -1:
            ax.scatter(*index, s=5, c = 'blue')
    plt.title('Final Model for 3D +/- J Model for OMCD')
    return plt.show()

#Showing the total change in the energy fo the lattice
def lattice_energy_change(graph_energy):
    plt.plot(graph_energy)
    plt.ylabel('Lattice Energy')
    plt.xlabel('Accepted Move Number')
    plt.title('Energy of the Lattice')
    return plt.show()

#shows the value of energy change at different values of d0
def change_in_energy(d0_when_energy_change, energy_change):
    plt.scatter(d0_when_energy_change, energy_change)
    plt.title('N = 800')
    plt.ylabel("Change in Energy")
    plt.xlabel("Value of d0 when energy changes")
    return plt.show()

def mean_energy_graph(d0_start_value, final_energy_per_spin):
    plt.scatter(d0_start_value, final_energy_per_spin)
    plt.ylabel('Final Lattice Energy')
    plt.xlabel('Sample Number')
    plt.title('Final Energy of the Lattice')
    return plt.show()

#above are old graphs that I might need later but not at this point 
def final_energy_per_spin_graph(d0_list, final_energy_averages):
    #I need to make these so they average many samples over a long time 
    plt.scatter(d0_list, final_energy_averages)
    plt.yscale('log')
    plt.errorbar(d0_list,final_energy_averages, yerr=final_energy_variances, linestyle="None")
    plt.title('Average Final Energies per Spin vs Initial Value of d0\n The Number of Samples Taken for Averages is 1000 \n Linear Deflation Schedule')
    plt.ylabel("Final Energy per Particle")
    plt.xlabel("d0")
    return plt.show()

def set_up():
    spin_glass_box_original = [[[np.random.choice([1, -1]) for i in range(nx)] for j in range(nx)] for k in range(nx)]
    spin_glass_box_original = np.asarray(spin_glass_box_original)
    spin_glass_box_medium = spin_glass_box_original.copy()
    J_matrix = [[[(np.random.choice([1, -1]), np.random.choice([1, -1]), np.random.choice([1, -1])) for ix in range(nx)] for jx in range(nx)] for kx in range(nx)]
    J_matrix = np.asarray(J_matrix)
    initial_total_energy = calc_total_energy(spin_glass_box_original, J_matrix)
    medium_total_energy = initial_total_energy
    graph_energy = [initial_total_energy]

    d0 = math.ceil((nx**3)/2) 
    d0_start_value.append(d0)

    acceptance = 0
    rejection = 0
    monte_carlo_step = round((nx**3)/d0)
    number_steps = 1000
    monte_carlo_steps = number_steps*monte_carlo_step

    return spin_glass_box_original, spin_glass_box_medium, J_matrix, initial_total_energy, medium_total_energy, graph_energy, d0, d0_start_value, acceptance, rejection, monte_carlo_steps

#just flattens the list f lists we get above 
def flatten(t):
    return [item for sublist in t for item in sublist]

def avg_accepted_moves_per_move_class_hist(sorted_keys, changes):
    plt.bar(sorted_keys, changes, color='g')
    plt.ylabel('Average Number of Accepted Moves')
    plt.xlabel('Move Class Value')
    plt.title('Average Number of Accepted Moves per Move Class\n Averaged over 100 Samples\n Starting Value of d0 = %i' %d0)
    return plt.show()

def avg_eng_per_move_class_hist(energy_dict):
    plt.bar(energy_dict.keys(), energy_dict.values(), color='g')
    plt.ylabel('Average Decrease in Energy')
    plt.xlabel('Move Class Value')
    plt.title('Average Energy Decrease per Move Class\n Averaged over 100 Samples\n Starting Value of d0 = 14')
    return plt.show()

def energy_decrease_each_move_class_hist(od):
    for i in od:
      #this gives the values at each point 
      list_i = od[i]
      counting_list = Counter(list_i)
      key = sorted_keys[i-1]
      plt.bar(counting_list.keys(), counting_list.values(), color='g')
      plt.ylabel('Number of Occurences')
      plt.xlabel('Decrease In Energy at This Move Class Value')
      plt.title('Occurences of Decreasing Energy Values at Move Class Value of d0 = %i' %key)
      plt.show()
      return plt.show()
  
#these next 2 lists are so I can try and recreate the graphs 
d0_when_energy_change = []
energy_change = []
d0_start_value = []

final_energy = []
samples = 100

input_list = [set_up() for i in range(samples)]

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.starmap(OMCD, input_list)
        energy_change = [x[2] for x in results]
        d0_when_energy_change = [x[3] for x in results]


#setting up the multprocessing so I can do more than 1 sample at a time. 

#creating a function to run omcd from d0 down to 1
# for i in range(samples):
#     set_up_values = set_up()
#     final_results = OMCD(spin_glass_box_original, spin_glass_box_medium, J_matrix, initial_total_energy, medium_total_energy, graph_energy, d0, d0_start_value, acceptance, rejection, number_rejections_this_round)
#     final_energy.append(final_results[1])

#     print('Just Finished Sample ' + str(i+1) + '/' + str(samples))

d0 = math.ceil((nx**3)/2) 
#dividing the values 
factor = (nx**3)
final_energy_per_spin = [x / factor for x in final_energy]

#turning lists into arrays so we can split them more easily 
d0_start_value = np.array(d0_start_value)
final_energy_per_spin = np.array(final_energy_per_spin)

#function that splits list up into chunks 
d0_chunks = np.array_split(d0_start_value, samples)
energy_chunks = np.array_split(final_energy_per_spin, samples)

#getting averages of each element 
final_energy_averages = np.mean(energy_chunks, axis=0)
#getting variances of each element 
final_energy_variances = np.var(energy_chunks, axis=0)

#average 
d0_list = d0_chunks[0]

#pulling results we nee for our histogram 

energy_change_avg = flatten(energy_change)
d0_when_energy_change_flat = flatten(d0_when_energy_change)

energy_dict = defaultdict(int)

for k, n in zip(d0_when_energy_change_flat, energy_change_avg):
  energy_dict[k] += abs(n)

energy_dict = {k: v / samples for total in (sum(energy_dict.values()),) for k, v in energy_dict.items()}

#turn histograms into function 
#historgram for average energy decrease per move class value 

zipped = zip(d0_when_energy_change_flat, energy_change_avg)

dict_1={}

for key,value in zipped:
    if key not in dict_1:
        dict_1[key]=[value]
    else:
        dict_1[key].append(value)

od = collections.OrderedDict(sorted(dict_1.items()))

sorted_keys = list(od.keys())

#histogram counting the number of accepted moves per move class value 
changes = []
for i in od:
  changes.append(len(od[i]))

changes = [x / samples for x in changes]

avg_accepted_moves_per_move_class_hist(sorted_keys, changes)
avg_eng_per_move_class_hist(energy_dict)


#histogram for occurences of energy change at each value of move class 


#printing the graphs for changes in energy 
# =============================================================================
# lattice_energy_change(graph_energy)
# change_in_energy(d0_when_energy_change, energy_change)
# mean_energy_graph(d0_start_value, final_energy_per_spin)
# =============================================================================

#final_config(spin_glass_box_medium)
#print('The final configuration of the the system is ' + str(final_results[0]))
#print('The initial total energy of the system is: ' + str(initial_total_energy))
#print('The final energy of this system is: ' + str(final_results[1]))
end = time.time()
print('This program took ' + str(end-start) + ' to run.')

#%%

flattened_energy_change = flatten(energy_change)
flattened_d0_when_energy_change = flatten(d0_when_energy_change)

zipped = zip(flattened_d0_when_energy_change, flattened_energy_change)
d = {}
for tuple in zipped:
    key,val = tuple
    d.setdefault(key, []).append(val)

lowest_bin_vals = []
for hist_val, values in d.items():
    if hist_val == 1:
        lowest_bin_vals.append(values)

second_lowest_bin_vals = []
for hist_val, values in d.items():
    if hist_val == 2:
        second_lowest_bin_vals.append(values)

lowest_bin_vals = flatten(lowest_bin_vals)
lowest_bin_vals_abs = []
for i in lowest_bin_vals:
    lowest_bin_vals_abs.append(abs(i))

second_lowest_bin_vals = flatten(second_lowest_bin_vals)
second_lowest_bin_vals_abs = []
for i in second_lowest_bin_vals:
    second_lowest_bin_vals_abs.append(abs(i))

nums = []
for i in range(len(lowest_bin_vals)):
    nums.append(i)
    
nums_2 = []
for i in range(len(second_lowest_bin_vals)):
    nums_2.append(i)
    
plt.scatter(nums, lowest_bin_vals_abs, s=3)
plt.ylabel('Energy Change')
plt.xlabel('Index of Accepted Move at d0 = 1')
plt.title('Examining the Energy Change Over \n Time at the Lowest Move-Class Value\n Averaged over %i Samples \n Linear Deflation Schedule' %samples)
plt.show()

plt.scatter(nums_2, second_lowest_bin_vals_abs, s=3)
plt.ylabel('Energy Change')
plt.xlabel('Index of Accepted Move at d0 = 2')
plt.title('Examining the Energy Change Over \n Time at the Second Lowest Move-Class Value\n Averaged over %i Samples \n Linear Deflation Schedule' %samples)
plt.show()