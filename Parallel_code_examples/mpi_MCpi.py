# run with mpirun -n 10 python mpi_MCpi.py

from mpi4py import MPI
import numpy as np
import math
import time

def compute_thing(x, y):
    thing = 0.0
    for j in range(1000):
        thing += math.sin(x + y) / math.cos(x - y)
    return thing

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    num_samples = 100000
    count = 0

    # Synchronize and start the timer
    comm.Barrier()
    start_time = time.time() if rank == 0 else None

    for i in range(num_samples // size):
        x, y = np.random.random(2)
        compute_thing(x, y)  # Additional computation
        if x**2 + y**2 <= 1.0:
            count += 1

    total_count = comm.reduce(count, op=MPI.SUM, root=0)

    if rank == 0:
        pi_estimate = 4.0 * total_count / num_samples
        end_time = time.time()
        print(f'Estimated Pi: {pi_estimate}')
        print(f'Parallel computation time: {end_time - start_time}')

if __name__ == '__main__':
    main()
