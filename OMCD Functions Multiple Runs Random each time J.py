# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:39:33 2021

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

#NOTE: These constants need to be changed based off scientific observation. We are not there though yet. 

#energy equation 
#H = -J*S_i*S_j, where J = +/-1 randomly 
J_values = [1, -1]

#this function to return the adjacent values of the point xyz
horiz_vert= lambda x,y,z:[(x,(y+1)%nx,z), ((x+1)%nx,y,z), (x,y,(z+1)%nx)]

#energy calculator, I think this works faster and is correct, old document in file name: omcd lattice (wrong maybe) 
#has the old code and I think is wrong. 
#it is also noteworthy that the old calculation was a lot slower than this lambda function 
def calc_total_energy(zero_matrix):
    total_energy = 0        
    for x in range(len(zero_matrix)):
        for y in range(len(zero_matrix)):
            for z in range(len(zero_matrix)):
                neighbours = horiz_vert(x, y, z)
                for i in neighbours:
                    if zero_matrix[i] == zero_matrix[x, y, z]:
                        total_energy += -1 
                    else: total_energy += -1*random.choice([1, -1])*zero_matrix[x, y, z]*zero_matrix[i]
    return(total_energy)              

#NOTE: Jij are random but fixed for the configurations 

def OMCD(d0, spin_glass_box_medium, medium_total_energy, acceptance, rejection, number_rejections_this_round):
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

        for k in flip_these:
            spin_glass_box_newest[k] = spin_glass_box_newest[k]*(-1)

# =============================================================================
#         #genertaing a matrix of ones
#         mask = np.random.choice([-1, 1], size=(nx, nx, nx), p=[d0/(nx**3), (nx**3 - d0)/(nx**3)]) 
#         #print('mask')
#         #print(mask)
# =============================================================================
# =============================================================================
#         
#         #using bitwise_xor function to flip the spins 
#         spin_glass_box_newest = mask*spin_glass_box_newest
#         #print(spin_glass_box_newest)
#         
# =============================================================================
        
        #calculating the new energy of the flipped spin glasses 
        newest_total_energy = calc_total_energy(spin_glass_box_newest)
      
        #functions says whether to accept (if lower) or reject (if higher) new system energy 
        if newest_total_energy <= medium_total_energy:
            energy_change.append(newest_total_energy - medium_total_energy)
            d0_when_energy_change.append(d0)
            medium_total_energy = newest_total_energy
            spin_glass_box_medium = spin_glass_box_newest
            acceptance += 1
            graph_energy.append(medium_total_energy)
            #print('This energy has been Accepted! Woooooooo!')
        else: 
            #print('Gonna have to reject this one, sorry bud.')
            rejection += 1
            number_rejections_this_round += 1
        
        #this if statement changes the number of spins we want to change but at the moment it shortens the length of the new spin system instead
        #read documentation on random.sample() to finish this part 
        if number_rejections_this_round == num_rejections_lower_d0:
            d0 = d0 - 1
            #print('To many rejections in a row, lets try lowering the number to see if that works. The new value of d0 is ' + str(d0))
            number_rejections_this_round = 0
            if d0 == 0:
                #print('d0 will not go any lower mate, gonna have to stop here.')
                break 
    
    #gives acceptance and rejection stats for this particular configuration 
    #print('The number of acceptances was ' + str(acceptance) + ' and the number of rejections was ' + str(rejection))
    #print('so our ratio of accepted to total number of attempted moves is ' + str(acceptance*100/(acceptance + rejection)) + '%')
    return spin_glass_box_medium, medium_total_energy


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
    plt.scatter(d0_list, final_energy_averages)
    plt.errorbar(d0_list,final_energy_averages, yerr=final_energy_variances, linestyle="None")
    plt.title('Average Final Energies per Spin vs Initial Value of d0\n The Number of Samples Taken for Averages is 400')
    plt.ylabel("Final Energy per Particle")
    plt.xlabel("d0")
    return plt.show()

#creating a 2d array of random ones and zeroes using a nested list 
spin_glass_box_original = [[[np.random.choice([1, -1]) for i in range(nx)] for j in range(nx)] for k in range(nx)]
print('Our initial spin glass configuration is: ' + str(spin_glass_box_original))

#converting this into a numpy array for the energy calculations to be easier
spin_glass_box_original = np.asarray(spin_glass_box_original)

initial_config(spin_glass_box_original)

#this will just be a place holder variable for when we get higher energy 
spin_glass_box_medium = spin_glass_box_original.copy()

initial_total_energy = calc_total_energy(spin_glass_box_original)
#print('The initial energy of the system is ' + str(initial_total_energy))

#energy for the medium spin glass list 
medium_total_energy = initial_total_energy 

#counting the acceptance and rejection ratio of the system 
acceptance = 0
rejection = 0
number_rejections_this_round = 0 

#these next 2 lists are so I can try and recreate the graphs 
d0_when_energy_change = []
energy_change = []

#gives a list of the final energys and the final configuration
#final_results = OMCD(d0, spin_glass_box_medium, medium_total_energy, acceptance, rejection, number_rejections_this_round)

final_energy = []
samples = 5
sample_list = []
for i in range(samples):
    sample_list.append(i)

d0_start_value = []

#creating a function to run omcd from d0 down to 1
for i in range(samples):
    spin_glass_box_original = [[[np.random.choice([1, -1]) for i in range(nx)] for j in range(nx)] for k in range(nx)]
    spin_glass_box_original = np.asarray(spin_glass_box_original)
    spin_glass_box_medium = spin_glass_box_original.copy()
    initial_total_energy = calc_total_energy(spin_glass_box_original)
    medium_total_energy = initial_total_energy
    graph_energy = [initial_total_energy]

    d0 = math.ceil((nx**3)/2) 
    for j in range(d0): #need to change it to d0-1 
        d0_start_value.append(d0)
        #counting the acceptance and rejection ratio of the system 
        acceptance = 0
        rejection = 0
        number_rejections_this_round = 0 
        #these next 2 lists are so I can try and recreate the graphs 
        d0_when_energy_change = []
        energy_change = []
        graph_energy = [initial_total_energy]
        final_results = OMCD(d0, spin_glass_box_medium, medium_total_energy, acceptance, rejection, number_rejections_this_round)
        d0 = d0-1
        final_energy.append(final_results[1])
    #print(final_results)
    print('Just Finished Sample ' + str(i+1) + '/' + str(samples))

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

# =============================================================================
# print(d0_list)
# print(final_energy_averages)
# =============================================================================


final_energy_per_spin_graph(d0_list, final_energy_averages)
# =============================================================================
# print(d0_chunks)
# print(energy_chunks)
# =============================================================================

final_config(final_results[0])

#printing the graphs for changes in energy 
# =============================================================================
# lattice_energy_change(graph_energy)
# change_in_energy(d0_when_energy_change, energy_change)
# mean_energy_graph(d0_start_value, final_energy_per_spin)
# =============================================================================

#final_config(spin_glass_box_medium)
print('The final configuration of the the system is ' + str(final_results[0]))
print('The initial total energy of the system is: ' + str(initial_total_energy))
print('The final energy of this system is: ' + str(final_results[1]))
end = time.time()
print('This program took ' + str(end-start) + ' to run.')