program mpi_monte_carlo_pi
    use mpi
    implicit none

    integer :: rank, size, ierr, i, j, count, total_count
    integer, parameter :: num_samples = 1000000
    double precision :: x, y, pi_estimate, start_time, end_time, thing
    double precision, external :: drand48

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    start_time = MPI_WTIME()

    count = 0
    do i = 1, num_samples / size
        call random_number(x)
        call random_number(y)
        thing = 0
        do j = 1,1000 ! add some stupid work
           thing = thing + sin(x+y)/cos(x-y)
        end do
        if (x**2 + y**2 <= 1.0d0) then
            count = count + 1
        end if
    end do

    call MPI_REDUCE(count, total_count, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        pi_estimate = 4.0d0 * total_count / num_samples
        print *, 'Estimated Pi:', pi_estimate
        end_time = MPI_WTIME()
        print *, 'Parallel computation time:', end_time - start_time
    end if

    call MPI_FINALIZE(ierr)
end program mpi_monte_carlo_pi
