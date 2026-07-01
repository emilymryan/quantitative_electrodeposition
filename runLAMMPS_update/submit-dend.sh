#! /bin/bash -l
#$ -l h_rt=200:00:00
#$ -P ryanlab
#$ -m bea
#$ -j y
#$ -N bv
#$ -o /projectnb/ryanlab/epenson/LAMMPS_AG_ELECTRODEPOSITION_NUCLEATION/results/final/$JOB_NAME.o$JOB_ID
#$ -e /projectnb/ryanlab/epenson/LAMMPS_AG_ELECTRODEPOSITION_NUCLEATION/results/final/$JOB_NAME.e$JOB_ID
#$ -pe mpi_16_tasks_per_node 32
module load openmpi/4.1.5
module load python3/3.6.5

lmp=/projectnb/ryanlab/epenson/mylammps_29Aug2024/src/lmp_mpi # use this lmp_mpi file to run LAMMPS

dname1=Apr02_1E-4RR_0.19V_${JOB_ID}
output=/projectnb/ryanlab/epenson/LAMMPS_AG_ELECTRODEPOSITION_NUCLEATION/results/${dname1}
dname=${output}
mkdir -p ${dname}

mpirun -np $NSLOTS ${lmp} -in input.lmp -var dname ${dname} # run input script in this folder 

exit

