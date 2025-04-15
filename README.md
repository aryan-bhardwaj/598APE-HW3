# 598APE-HW3

This repository contains code for homework 3 of 598APE.

This assignment is relatively simple in comparison to HW1 and HW2 to ensure you have enough time to work on the course project.

In particular, this repository is an implementation of an n-body simulator.

To compile the program run:
```bash
make -j
```

To clean existing build artifacts run:
```bash
make clean
```

This program assumes the following are installed on your machine:
* A working C compiler (g++ is assumed in the Makefile)
* make

The nbody program is a classic physics simulation whose exact results are unable to be solved for exactly through integration.

Here we implement a simple time evolution where each iteration advances the simulation one unit of time, according to Newton's law of gravitation.

Once compiled, one can call the nbody program as follows, where nplanets is the number of randomly generated planets for the simulation, and timesteps denotes how long to run the simulation for:
```bash
./main.exe <nplanets> <timesteps>
```
In order to run the artifact, first run `git checkout test`. Then, you will have to checkout specific commit hashes to test specific optimizations. Here are the commit hashes of the major commits to reproduce my results. 
- dd69847231ebcf997fa41c037256b68f1750ced5 (added compiler flags)
- f9ec6cf5c8472c0690ca2a09d8d86a68811ea0ec (added double buffering)
- 1de2ee27d80b19c3ce075e464c641ce23740a2a7 (added OpenMP parallelization)
- 98710c0e45ed097443164ac0e8c982ac93547c10 (added inlining to random number functions)
- c1b54ba2a261fadf3c8a79644e9b0814d618ee37 (replaced AoS with SoA, used alignas(64) for struct, and added vectorization)

Here are the test cases showed in the report. 
Test Case 1: 1000 planets, 5000 timesteps
```bash
./main.exe 1000 5000
```
Test Case 2a: 8 planets, 1000000 timesteps
```bash
./main.exe 8 1000000
```
Test Case 2b: 8 planets, 10000000 timesteps
```bash
./main.exe 8 10000000
```
Test Case 2c: 8 planets, 100000000 timesteps
```bash
./main.exe 8 100000000
```
