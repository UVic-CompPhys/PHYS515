# Import the MPI module from mpi4py
from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD

# Get the rank of the process
rank = comm.Get_rank()

# Print the rank of each process
print(f"Hello from the process {rank}")
