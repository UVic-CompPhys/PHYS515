program parallel_sum
    use mpi
    implicit none

    integer :: rank, size, ierr, i
    integer, parameter :: n = 1000000
    double precision :: local_sum, total_sum, start_time, end_time

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    start_time = MPI_WTIME()

    local_sum = 0.0
    do i = rank + 1, n, size
        local_sum = local_sum + i
    end do

    call MPI_REDUCE(local_sum, total_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        print *, 'Total sum:', total_sum
        end_time = MPI_WTIME()
        print *, 'Parallel computation time:', end_time - start_time
    end if

    call MPI_FINALIZE(ierr)
end program parallel_sum
