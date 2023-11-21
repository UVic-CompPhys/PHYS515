program sequential_sum
    implicit none

    integer :: i
    integer, parameter :: n = 1000000
    double precision :: total_sum, start_time, end_time

    start_time = MPI_WTIME()
    total_sum = 0.0

    do i = 1, n
        total_sum = total_sum + i
    end do

    end_time = MPI_WTIME()
    print *, 'Total sum:', total_sum
    print *, 'Sequential computation time:', end_time - start_time
end program sequential_sum
