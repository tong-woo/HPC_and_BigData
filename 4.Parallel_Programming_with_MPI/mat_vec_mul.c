/* File:     mat_vec_mul.c
 * Author:   HPC_Group6: Tong Wu | Leonardo Kuffó | Zhaolin Fang
 *
 * Purpose:  Implement parallel matrix-vector multiplication based on MPI
 *           using one-dimensional array to store them. The matrix is 
 *           distributed by block rows. The vector is distributed by block 
 *           column
 *
 * Compile:  gcc -o mat_vec_mul mat_vec_mul.c(for now) || mpiCC...(later)
 * Run:      ./mat_vec_mult(for now) || mpirun ./mat_vec_mul(later)
 *
 * Input:    1.Dimensions of the matrix and vector (n = number of rows
 *           = number of columns)
 *           2.n x n matrix A
 *           3.n-dimensional vector x
 * Output:   Product value result = A[row]*x(for now) || Product vector y = Ax(later)
 *
 * Errors:   TBA
 *
 * Notes:    1. Try to add error handling strategy in each function as required
 */

#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 
#include <omp.h>

/*---------implicit declaration of functions not allowed(C99)--------*/
void Build_matrix(double A[], int n);
void Build_vector(double x[], int N);
void Print_matrix(char name[], double A[], int n);
void Print_vector(char name[], double vec[], int n);
double row_vec_mul(double * A, int row, double * x,int N);
void run_as_master(
    int ITERATIONS_R,
    int worker_count, 
    int DIMENSION_SIZE, 
    double *VECTOR_RESULT, 
    double *VECTOR_V, 
    double *MATRIX
);
void run_as_worker(int DIMENSION);

/* TAGs for MPI communication */
int TAG_DIMENSION = 1;
int TAG_VECTOR_V = 2;
int TAG_RESULT = 3;
int TAG_MATRIX_BLOCK = 4;

/*--------------------------------------------------------------------*/
int main(int argc, char *argv[]) {

    /* Start up MPI */
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Dimension N
    int DIMENSION_SIZE = 30000;
    // Iterations R
    int ITERATIONS_R = 300;

    // Seed to generate random nums
    srand((unsigned)time(NULL)); 

    const bool am_master = 0 == rank;

    if (am_master) {
        // Build random Matrix M and Vector V
        double MATRIX[(long)DIMENSION_SIZE * (long)DIMENSION_SIZE]; // long conversions are used to prevent integer multiplication overflow
        double VECTOR_V[DIMENSION_SIZE];
        double VECTOR_RESULT[DIMENSION_SIZE];
        Build_matrix(MATRIX, DIMENSION_SIZE);   
        Build_vector(VECTOR_V, DIMENSION_SIZE);  

        int workers = size - 1;
        printf("Running as master with %d workers\n", workers);

        const double start = MPI_Wtime();
        run_as_master(
            ITERATIONS_R
            workers, 
            DIMENSION_SIZE,
            VECTOR_RESULT,
            VECTOR_V,
            MATRIX
        );
        const double finish = MPI_Wtime();
        printf("Stopped as master. This took %.4f seconds\n", finish-start);
    } else {
        printf("Running as worker %d\n", rank);
        run_as_worker(
            DIMENSION_SIZE
        );
        printf("Stopped as worker %d\n", rank);
    }

    MPI_Finalize();
    return 0;
}  /* main */

/*-------------------------------------------------------------------
 * Function:  Generate random double
 * Purpose:   Generate a random double value between two double numbers
 * In args:   fMin: Minimum possible value
              fMax: Maximum possible value
 * Out args:  double f: a random double value between fMin and fMax
 */
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


/*-------------------------------------------------------------------
 * Function:  Build_matrix
 * Purpose:   Build the matrix acccording to the input size N with 
 *            random numbers between 0 and 100
 * In args:   N: local number of cols in A
 * Out args:  A: the local matrix
 * 
 * Errors:    NA
 * Note:      1.To debug code with something predictable, maybe use 
 *            an identity matrix instaed of this function
 */
void Build_matrix(
      double A[]  /* out */, 
      int    N    /* in  */) {
   int i, j;

   for (i = 0; i < N; i++)
      for (j = 0; j < N; j++) 
         A[i*N + j] = fRand(0.0, 100.0);
}  /* Build_matrix */


/*-------------------------------------------------------------------
 * Function:  Build_vector
 * Purpose:   Build the vector acccording to the input size N with 
 *            random double numbers between 0 and 100
 * In args:   N: local number of cols in A
 * Out args:  x[]: the local vector
 *
 * Errors:    NA
 * Note:      1.To debug code with something predictable, maybe use 
 *            an identity vector instaed of this function
 */
void Build_vector(
      double x[] /* out */, 
      int    N   /* in  */) {
   int i;

   for (i = 0; i < N; i++)
      x[i] = fRand(0.0, 100.0);
}  /* Build_vector */


/*-------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print a matrix distributed by row-major order to stdout
 * In args:   name: name of matrix
 *            A:    the matrix
 *            n:    number of cols
 * Out args:  NA
 * 
 * Errors:    NA
 * Notes:     NA
 */
void Print_matrix(
    char      name[]    /* in */,
    double    A[]       /* in */, 
    int       n         /* in */) {
        int i, j;
        printf("\nThe matrix %s\n", name);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }
}  /* Print_matrix */


/*-------------------------------------------------------------------
 * Function:  Print_vector
 * Purpose:   Print a vector with a column distribution
 * In args:   name: name of vector
 *            vec:  the vector array
 *            n:    global number of components
 * Out args:  NA
 * 
 * Errors:    NA
 * Notes:     NA
 */
void Print_vector(
    char      name[] /* in */, 
    double    vec[]  /* in */, 
    int       n      /* in */) {
        int i;
        printf("\nThe vector %s\n", name);
        for (i = 0; i < n; i++){
            printf("%f ", vec[i]);
        }
        printf("\n");
}  /* Print_vector */       



/*-------------------------------------------------------------------
 * Function:  row_vec_mul
 * Purpose:   Multiply a row vector of matrix A and a vector x. The matrix 
 *            is distributed by row blocks and the vectors are distributed 
 *            by blocks of column
 * In args:   A:   matrix A
 *            row: the row index of matrix (e.g. 0,1,2...N-1)
 *            x:   vector x
 *            N:   number of columns
 * out args:  res: stores the duble value of vector product
 * 
 * Errors:    1.if the row index >= N or less then 0, then exit and print a 
 *            error message
 *            2.if malloc of local storage on any process fails, all
 *            processes quit.            
 * Notes:     NA
 */
double row_vec_mul(
    double    * A  /* in  */,
    int       row  /* in  */, 
    double    * x  /* in  */, 
    int       N    /* in  */){
        if(row >= N || row < 0){
            printf("The row index is invalid(row_idx > N)");
            exit(1);
        }
        int j;
        double res = 0.0; /* out  */
        for (j = 0; j < N; j++){
            res += A[row*N+j]*x[j];
        }
    return res;
}


/**
 * The code to run as worker: receive jobs, execute them, and terminate when told to.
 */
void run_as_worker(
    int DIMENSION_SIZE
) {
    MPI_Status status;
    int block_size = 0;
    bool have_received_matrix = false;
    double * MATRIX_BLOCK;
    double * result;
    double * VECTOR_V = (double *) malloc (DIMENSION_SIZE * sizeof(double));
    while (true){
        MPI_Bcast(
            VECTOR_V, 
            DIMENSION_SIZE, 
            MPI_DOUBLE,  
            0, 
            MPI_COMM_WORLD
        );

        if (!have_received_matrix){
            MPI_Recv(
                &block_size,
                1,
                MPI_INT, 0, 
                TAG_DIMENSION, MPI_COMM_WORLD,
                &status
            );

            MATRIX_BLOCK = (double *) malloc( DIMENSION_SIZE * block_size * sizeof(double));
            result = (double *) malloc (block_size * sizeof(double));

            MPI_Recv(
                MATRIX_BLOCK, 
                DIMENSION_SIZE * block_size, 
                MPI_DOUBLE, 
                0, TAG_MATRIX_BLOCK,
                MPI_COMM_WORLD, &status
            );
            have_received_matrix = true;
        }

        // Perform matrix multiplacion
        for (int j = 0; j < block_size; j++){
            result[j] = row_vec_mul(MATRIX_BLOCK, j, VECTOR_V, DIMENSION_SIZE);
        }

        MPI_Send(
            result, 
            block_size, 
            MPI_DOUBLE, 0, 
            TAG_RESULT, 
            MPI_COMM_WORLD
        );
    }
}



/**
 * The code to run as master: send jobs to the workers, and await their replies.
 * There are `worker_count` workers, numbered from 1 up to and including `worker_count`.
 *
 * @param worker_count The number of workers
 * @param startval  The first value to examine.
 * @param nval The number of values to examine.
 * @return The number of values in the specified range.
 */
void run_as_master(
    int ITERATIONS_R,
    int worker_count, 
    int DIMENSION_SIZE, 
    double *VECTOR_RESULT, 
    double *VECTOR_V, 
    double *MATRIX
) {
    bool is_last_iteration = false;
    int *worker_block_start = (int*) malloc(sizeof(int) * worker_count);
    int *worker_block_sizes = (int*) malloc(sizeof(int) * worker_count);
    int block_size = ceil(DIMENSION_SIZE / worker_count);

    for(int ITERATION = 0; ITERATION < ITERATIONS_R; ITERATION++) {

            MPI_Bcast(
                VECTOR_V, 
                DIMENSION_SIZE, 
                MPI_DOUBLE,  
                0, 
                MPI_COMM_WORLD
            );

            int active_workers = 0, dimensions_sent = 0;
            for (int worker = 1; worker <= worker_count && dimensions_sent < DIMENSION_SIZE; worker++) {
                if (ITERATION > 0){ // Relevant information has already been sent to this workers
                    active_workers = worker_count;
                    break;
                }
                int worker_block_size = block_size;
                if ((dimensions_sent + block_size) > DIMENSION_SIZE){
                    worker_block_size = DIMENSION_SIZE - dimensions_sent;
                }

                MPI_Send(
                    &worker_block_size, 
                    1, 
                    MPI_INT, worker, 
                    TAG_DIMENSION, MPI_COMM_WORLD
                );

                MPI_Send(
                    MATRIX + dimensions_sent * DIMENSION_SIZE,
                    DIMENSION_SIZE * worker_block_size, 
                    MPI_DOUBLE, worker, 
                    TAG_MATRIX_BLOCK, MPI_COMM_WORLD
                );

                worker_block_start[worker] = dimensions_sent;
                worker_block_sizes[worker] = worker_block_size;
                active_workers++;
                dimensions_sent += worker_block_size;
            }
            while (active_workers > 0) {
                int worker;
                double * result = (double *) malloc(sizeof(double) * block_size);

                MPI_Status status;
                MPI_Recv(
                    result, 
                    block_size, 
                    MPI_DOUBLE, MPI_ANY_SOURCE, 
                    TAG_RESULT, MPI_COMM_WORLD, &status
                );
                worker = status.MPI_SOURCE;

                memcpy(
                    VECTOR_RESULT + worker_block_start[worker], 
                    result, 
                    sizeof(double) * worker_block_sizes[worker]
                );
                active_workers--;
            }

            memcpy(
                VECTOR_V, 
                VECTOR_RESULT, 
                sizeof(double) * DIMENSION_SIZE
            );
    }
}
