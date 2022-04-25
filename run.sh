#!/bin/bash
#SBATCH --job-name="horovod_tensorflow_mnist"
#SBATCH --output="horovod_tensorflow_mnist_%j.out"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --cores-per-socket=20
#SBATCH --threads-per-core=4
#SBATCH --sockets-per-node=2
#SBATCH --mem-per-cpu=1200
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:4
#SBATCH --time=4:00:00

mpirun -n 3 python Unet.py