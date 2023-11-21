program hybrid_monte_carlo_pi
    use mpi
    use omp_lib
    implicit none

    integer :: rank, size, ierr, i, j, local_count, total_count, num_samples_per_process
    integer, parameter :: num_samples = 1000000
    double precision :: x, y, pi_estimate, start_time, end_time, thing

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    ! Splitting the total samples among available MPI processes
    num_samples_per_process = num_samples / size

    ! Initializing OpenMP
    call omp_set_num_threads(omp_get_max_threads())
    local_count = 0

    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    start_time = MPI_WTIME()

    ! Parallel region using OpenMP
    !$omp parallel private(x, y, i) reduction(+:local_count)
    ! note that we are not seeding the random generator correctly as we should
    !$omp do
    do i = 1, num_samples_per_process
        call random_number(x)
        call random_number(y)
        thing = 0
        do j = 1,1000 ! add some stupid work
           thing = thing + sin(x+y)/cos(x-y)
        end do
        if (x**2 + y**2 <= 1.0d0) then
            local_count = local_count + 1
        end if
    end do
    !$omp end do
    !$omp end parallel

    ! MPI reduction to sum up the counts from all processes
    call MPI_REDUCE(local_count, total_count, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        pi_estimate = 4.0d0 * total_count / num_samples
        print *, 'Estimated Pi:', pi_estimate
        end_time = MPI_WTIME()
        print *, 'Hybrid computation time:', end_time - start_time
    end if

    call MPI_FINALIZE(ierr)
end program hybrid_monte_carlo_pi
