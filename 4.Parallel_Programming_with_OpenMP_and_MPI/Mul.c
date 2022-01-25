#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


void Check_error(int local_ok, char fname[], char message[], 
      MPI_Comm comm);
void Get_dims(int* m_p, int* local_m_p, int* n_p, int* local_n_p,
      int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_arrays(double** local_A_pp, double** local_x_pp, 
      double** local_y_pp, int local_m, int n, int local_n, 
      MPI_Comm comm);
void Read_matrix(char prompt[], double local_A[], int m, int local_m, 
      int n, int my_rank, MPI_Comm comm);
void Read_vector(char prompt[], double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm);
void Print_matrix(char title[], double local_A[], int m, int local_m, 
      int n, int my_rank, MPI_Comm comm);
void Print_vector(char title[], double local_vec[], int n,
      int local_n, int my_rank, MPI_Comm comm);
void Mat_vect_mult(double local_A[], double local_x[], 
      double local_y[], int local_m, int n, int local_n, 
      MPI_Comm comm);

/*-------------------------------------------------------------------
 * Function:  main
 * Purpose:   execute program  y = A*x
 * MPI funcs: MPI_Comm_size: determine the size of each group
 *            MPI_Comm_ramk: Determines the rank of the calling process in the communicator
 *            MPI_Barrier: Blocks until all processes in the communicator have reached this routine.
 *            MPI_Init: Initialize the MPI execution environment
 *            MPI_Reduce: Reduces values on all processes to a single value
 */
int main(void) {
  double* local_A;
  double* local_x;
  double* local_y;
  int m, local_m, n, local_n;
  int my_rank, comm_sz;
  MPI_Comm comm;
  double start, finish, loc_elapsed, elapsed;

  MPI_Init(NULL, NULL);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_sz); 
  MPI_Comm_rank(comm, &my_rank);  

  Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);
  Allocate_arrays(&local_A, &local_x, &local_y, local_m, n, local_n, comm);
  Read_matrix("A", local_A, m, local_m, n, my_rank, comm);
#  ifdef DEBUG
  Print_matrix("A", local_A, m, local_m, n, my_rank, comm);
#  endif
  Read_vector("x", local_x, n, local_n, my_rank, comm);
#  ifdef DEBUG
  Print_vector("x", local_x, n, local_n, my_rank, comm);
#  endif

  MPI_Barrier(comm);
  start = MPI_Wtime();

  Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, comm);

  finish = MPI_Wtime();
  loc_elapsed = finish-start;
  MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  Print_vector("y", local_y, m, local_m, my_rank, comm);
  if (my_rank == 0)
    printf("Elapsed time = %e\n", elapsed);
  printf("Execution time = %e\n", loc_elapsed);
  free(local_A);
  free(local_x);
  free(local_y);
  MPI_Finalize();
  return 0;
}  /* main */
