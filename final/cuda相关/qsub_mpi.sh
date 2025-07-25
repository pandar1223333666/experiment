#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=2:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)


for node in $NODES; do
    scp master_ubss1:/home/${USER}/guess/main_mpi ${node}:/home/${USER} 1>&2
    scp -r master_ubss1:/home/${USER}/guess/files ${node}:/home/${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np 16 -machinefile $PBS_NODEFILE /home/${USER}/main_mpi

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/guess/ 2>&1

