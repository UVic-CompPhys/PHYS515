

### Exploring Parallel Computing and Amdahl's Law



### **Task 1: Strong Scaling Test with OpenMP**
**Objective:** Perform a strong scaling test for a computational problem using Fortran and OpenMP.

1. Choose a simple computational task like solving the **1D heat diffusion equation**: A 1D rod of length $ L = 1.0 $with thermal diffusivity $\alpha = 0.01 $. Initial temperature distribution: $T(x, 0) = \sin(\pi x)$  representing an initial heat wave. Boundary conditions: $T(0, t) = 0, \quad T(L, t) = 0.$ Plot the results to make sure they make sense.
2. Run the program for a fixed problem size (e.g.,  $n = 10^6 $ with different numbers of threads (e.g., 1, 2, 4, 8, 16).
3. Record and plot the speedup vs. the number of threads.
4. Analyze the results in terms of Amdahl’s Law:
   $$
   S_p = \frac{1}{(1 - f) + \frac{f}{p}},
   $$
   where \( f \) is the parallelizable fraction and \( p \) is the number of threads.

**Deliverables:** A plot of speedup vs. threads, with a discussion on where diminishing returns begin to appear.

### Task 2: Weak Scaling Test with MPI

**Objective:** Perform a weak scaling test for a distributed memory task using MPI.

1. Write a Fortan MPI program to implement a simple **domain decomposition** problem (e.g., splitting a 1D array and computing its sum or some more complicated artificial work).
2. Scale the problem size proportionally to the number of MPI processes (e.g., 1 process handles \( n = 10^6 \), 2 processes handle \( n = 2 \times 10^6 \), etc.).
3. Measure the runtime for different process counts (e.g., 1, 2, 4, 8) and verify that the runtime remains constant (ideal weak scaling).

**Deliverables:** A plot of runtime vs. number of processes, with a discussion on any deviations from ideal weak scaling.