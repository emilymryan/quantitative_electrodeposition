# Calculation of Current Density
import numpy as np
import pandas as pd

job = 'Sept08_0.25V_current_density_230335' # update based on your job
fname = '/projectnb/ryanlab/epenson/LAMMPS_AG_ELECTRODEPOSITION_NUCLEATION/results/current_density/' + job

# information for input files
Nfreq = 1
totDump = 10
time = 0
dt = 2.80017921146952e-06 # change based on simulation

# Current density parameters
N = 1 # oxidation state for Ag
mAg = 107.9 # atomic mass of Ag
Faraday = 96485
L = 10 # um

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

##Create array to hold dmM and time dmM[ionicflux,time]
matrix = np.zeros((totDump,3),dtype=float)


#loop through X  dumps that start at 0 (subtract t=0 from loop)
for i in range(0,totDump):
    matrix[i,1] = time
    time = time + (Nfreq*dt)
    num = i*Nfreq
    fid = fname + "/dump." + str(num) + ".dat"
    data = np.loadtxt(fid, skiprows=9, usecols=9) # dm/dt double check placement of dmM value based on your dump files!
    dmM = np.sum(data)*pow(10,-15) # conversion from (ug/uL/s * mass/rho)  to (g/um3/s * mass/rho) (1x10^9 um3 = 1 uL)
    currentdensity = (dmM*Faraday)/(mAg*pow(L,2)) # A/um2
    matrix[i,2] = currentdensity
    matrix[i,0] = num
    #print("Current Density: " + str(currentdensity) + "\n time: " + str(time) + "\n")
    print("Print Out: " + str(num) + "\n time: " + str(time) + "\n Current Density: " + str(currentdensity) + "\n")
    #print("Print Out: " + str(num) + "\n time: " + str(time) + "\n sumdmM: " + str(dmM) + "\n")
#Save currentdensity data to csv
CurrentDensityData = pd.DataFrame(matrix)
#CurrentDensityData.to_csv('Sept08_0.22V_current_density_230692.csv',index=False) #Change Job



