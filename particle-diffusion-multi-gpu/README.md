# Particle Motion Simulation on Multiple GPUs

This repository hosts implementations of particle motion simulation using different programming models across multiple GPUs, using CUDA, OpenMP, and MPI. The aim of these implementations is to demonstrate and evaluate the performance and efficiency of simulating particle motion, using different scheduling approaches and granularity.

## Overview

This project explores different computational strategies to perform these simulations: CUDA, OpenMP and MPI.

The focus is on analyzing how different parallel computing techniques can enhance the simulation's performance and scalability.

## Installation

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes:

```bash
# Clone this repository
https://github.com/dikshant293/HeCBench.git

# Go into the repository
cd cd HeCBench/particle-diffusion-multi-gpu/

# For CUDA implementations, ensure you have the CUDA toolkit installed
# For OpenMP implementations, ensure your compiler supports OpenMP
# For MPI implementations, install an MPI library like MPICH or OpenMPI
```
## Compilers
* CUDA - nvc++
* OpenMP - LLVM clang++
* MPI - mpich (openmpi)

## Running

All implementations are accompanied with `Makefile`s for easy running and testing. Running the `make` command compiles the code into an executable. For simulating N particle for M iterations with g granularity:

```bash
./<executableName> <M> <g> <N>
```

To reproduce data provided in the paper run:
* CUDA: `make gran-test`
* OpenMP: `make gran-test`
* MPI: `make test` for MPI+CUDA and `make test API=mpi-omp SUBCOMPILER=clang` for MPI+OMP