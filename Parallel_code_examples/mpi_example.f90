program hello_mpi
    use mpi
    implicit none

    integer :: rank, size, ierr

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

    print *, 'Hello world from process ', rank, ' of ', size

    call MPI_FINALIZE(ierr)
end program hello_mpi
