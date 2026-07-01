# Calculate Density as a Function of Y-Distance
# Feb 10 2025
# Elyssa Penson
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

job = 'Dec28_1E-4RR_0.22V_3.6e-11_surf_energy_2505909'
fname = '/projectnb/ryanlab/epenson/LAMMPS_AG_ELECTRODEPOSITION_NUCLEATION/results/' + job

Nfreq = 14284 #7142 #(use every other dump file)
totDump = 42
dumpT = 0.04
time = 0
totAtoms = 1440000
length = np.zeros((totAtoms,1),dtype=float)
Lx = 10.0

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

data = np.zeros((totDump,4),dtype=float)

#loop through totDumps that start at 0

for i in range(0,totDump):
    time = time + dumpT
    num = i*Nfreq
    fid = fname + "/dump." + str(num) + ".dat"
    particle_type = np.loadtxt(fid, skiprows=9, usecols=1)
    #x = np.loadtxt(fid, skiprows=9, usecols=2)
    y = np.loadtxt(fid, skiprows=9, usecols=3)
    #mask = (x >= 2.5) & (x <= 3.5)
    #x = x[mask]
    #y = y[mask]
    particle_type = np.array(particle_type) #[mask]
    density = [1 if int(j) == 2 else 0 for j in particle_type]
    bin_edges = np.array([0 , 0.06, 0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 0.96, 1.06, 1.16, 1.26, 1.36, 1.46])#, 1.56, 1.66, 1.76, 1.86, 1.96, 2.06, 2.16])
    bin_widths = np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    bin_stats = binned_statistic(y, density,'mean', bins = bin_edges)
    print("density:", bin_stats.statistic)
    #print("bin edges:", bin_stats.bin_edges)
    print("bin centers:", bin_centers)
    avg_density = bin_stats.statistic
    dist_from_anode = list(bin_centers)
    dist_from_anode[0] -= 0.03 # dist from anode is 0 for the first y bin
    
for i in range(1, len(dist_from_anode)):
    dist_from_anode[i] = dist_from_anode[i] - 0.06
    #print("dist_from_anode:", dist_from_anode)

#plt.plot(dist_from_anode, avg_density, label='Reaction Rate = 1E-6 (\u00B5m/s)', color='red') # change label    
plt.plot(dist_from_anode, avg_density, label='Surface Energy = 3.6e-11 (J/\u00B5m\u00B2)', color='red') # change label
plt.xlim(left=0)
#plt.xlim(right=0.5)
plt.ylim(bottom=0)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.legend()
plt.xlabel("Distance from Platinum Electrode (\u00b5m)")
plt.ylabel("Density of Deposited Silver")
# Density is the number of solid silver particles / total number of particles
plt.savefig('Dec28_1E-4RR_0.22V_3.6e-11_surf_energy_2505909_1.67s.png') # Change name of figure
plt.show()

#Save Y-distance and density to csv
Data = pd.DataFrame({'Y': dist_from_anode,'Dendrite Density': avg_density})
#Data.to_csv('Dec28_1E-4RR_0.22V_3.6e-11_surf_energy_2505909_1.67s.csv',index=False) # Change name of csv file


    
 
