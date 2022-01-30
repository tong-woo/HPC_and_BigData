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
void Set_dims(int* n_p , int* N_p );
void Allocate_dynamic_arrays(double** A_pp, double** x_pp, 
     double** y_pp, int N);
void Build_matrix(double A[], int n);
void Build_vector(double x[], int N);
void Print_matrix(char name[], double A[], int n);
void Print_vector(char name[], double vec[], int n);
void Mat_vec_mul(double A[], double x[], 
      double y[], int N);
double row_vec_mul(double A[], int row, double x[],int N);
double *run_as_master(
    int worker_count, 
    int DIMENSION_SIZE, 
    double *VECTOR_RESULT, 
    double *VECTOR_V, 
    double *MATRIX,
    bool last_iteration
);
void run_as_worker(int DIMENSION);

int TAG_DIMENSION = 1;
int TAG_VECTOR_V = 2;
int TAG_RESULT = 3;
int TAG_MATRIX_BLOCK = 4;


/*--------------------------------------------------------------------*/
int main(int argc, char *argv[]) {

    int rank;
    int size;

    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int DIMENSION_SIZE = 10000; // N
    double MATRIX[DIMENSION_SIZE][DIMENSION_SIZE];
    double VECTOR_V[DIMENSION_SIZE];
    double VECTOR_RESULT[DIMENSION_SIZE];

    srand((unsigned)time(NULL)); //set seed to generate random nums

    const bool am_master = 0 == rank;

    printf("Running");
    if (am_master) {
        printf("Running as master");
        Build_matrix((double *)MATRIX, DIMENSION_SIZE);
        printf("Matrix built");       
        Build_vector(VECTOR_V, DIMENSION_SIZE);            
        int workers = size - 1;
        printf("Running as master with %d workers\n", workers);
        const double start = MPI_Wtime();
        int R = 2;
        bool is_last_iteration = false;
        for(int i=1; i<=R; i++) {
                if(i==R) is_last_iteration = true;
                double * VECTOR_R = run_as_master(
                    workers, 
                    DIMENSION_SIZE,
                    VECTOR_RESULT,
                    VECTOR_V,
                    (double*) MATRIX,
                    is_last_iteration
                );
                //VECTOR_V = VECTOR_R;
                // Print_vector("R", VECTOR_R, DIMENSION_SIZE);
        }
        const double finish = MPI_Wtime();
        printf("Stopped as master. This took %.4f seconds\n", finish-start);
    } else {
        printf("Running as worker %d\n", rank);
        run_as_worker(DIMENSION_SIZE);
        printf("Stopped as worker %d\n", rank);
    }

    MPI_Finalize();

    free(MATRIX);
    free(VECTOR_V);
    free(VECTOR_RESULT);
    return 0;
}  /* main */

/*--------------------------------------------------------------------*/
int main2(void) {
   double* A;
   double* x;
   double* y;
   int n, N;
   double result;
   int row_idx = 0;                  
   Set_dims(&n, &N);                 //Hover on the functions to see details in VScode
   Allocate_dynamic_arrays(&A, &x, &y, N);
   srand((unsigned)time(NULL));      //set seed to generate random nums
   Build_matrix(A, N);               //Matrix array stored in A
   // Print_matrix("A", A, N);  
   Build_vector(x, N);               //Vector array stored in x
   // Print_vector("x", x, N);
   /*uncomment below to verify if the first element of y vector is the same as result value*/
   // Mat_vec_mul(A, x, y, N);
   // Print_vector("y", y, N);
   result = row_vec_mul(A, row_idx, x, N); // Now the product of vectors(A[row_idx] and X) has been stored in result
   printf("\nThe product of vector and row %d of Matrix \n%f", row_idx+1, result);
   free(A);
   free(x);
   free(y);
   return 0;
}  /* main */

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/**
 * Given a value to send and a worker to send the value to, send
 * a message ordering the worker to compute whether the given value
 * is prime. The special value '0' means that the worker should quit.
 * The worker will send a message to the master with the verdict.
 *
 * @param worker  The worker to send the message to.
 * @param val The value to examine.
 */
void send_work_command(int worker, double* matrix_and_vector, int data_to_send_dimension) {
    // printf("send_work_command: worker=%d val=%lu\n", worker, val);
    MPI_Send(matrix_and_vector, data_to_send_dimension, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
}

void turn_off_worker(int worker) {
    double dummy = 0.0;
    MPI_Send(&dummy, 1, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
}

/**
 * Given a result to send, send a message telling the master that
 * the value it sent is prime or not. Note that the value for which the
 * result has been computed is not sent, since nobody cares about it.
 *
 * @param result The result.
 */
void send_result(double result) {
    
}

/**
 * Wait for the next result from a worker.
 *
 * @param worker A pointer to the variable that will hold the worker that sent the result.
 * @param result A pointer to the variable that will hold the result.
 */
void await_result(int *worker, double *result, int worker_block_size) {

}

/*-------------------------------------------------------------------
 * Function:  Set_dims
 * Purpose:   Set the dimensions of the matrix and the vectors from stdin.
 * In args:   NA
 * Out args:  *n_p: global number of cols of A and rows of x
 *            *N_p: local number of cols of A and rows of x
 *
 * Errors:    1.if either m or n isn't positive the
 *            program prints an error message and quits.
 * Note:      NA
 */
void Set_dims(
    int*      n_p  /* out */,
    int*      N_p  /* out */) {
        printf("Enter the size of matrix and vector\n");
        scanf("%d", n_p);
        if(*n_p<0){
            printf("invalid input");
            exit(-1);
        }
        *N_p = *n_p;
}  /* Set_dims */


/*-------------------------------------------------------------------
 * Function:   Allocate_dynamic_arrays
 * Purpose:    Allocate memory for A, x, and y, they are 
 *             using dynamic arrays
 * In args:    N:    number of rows in x
 * Out args:   A_pp: pointer to memory adress for matrix 
 *             x_pp: pointer to memory adress for x 
 *             y_pp: pointer to memory adress for y 
 * 
 * Errors:     if a malloc fails, the program prints a message and all
 *             processes quit
 * Note:       NA
 */
void Allocate_dynamic_arrays(
    double**  A_pp  /* out */, 
    double**  x_pp  /* out */, 
    double**  y_pp  /* out */,    
    int       N     /* in  */) {

        *A_pp = (double *) malloc(N*N*sizeof(double));
        *x_pp = (double *) malloc(N*sizeof(double));
        *y_pp = (double *) malloc(N*sizeof(double));
        if (!A_pp || !x_pp || !y_pp) {
        fprintf(stderr, "%s: memory exhausted\n", "mat_vec_mul.c");
        exit(1);
    }
}  /* Allocate_dynamic_arrays */


/*-------------------------------------------------------------------
 * Function:  Build_matrix
 * Purpose:   Build the matrix acccording to the input size N with 
 *            random numbers
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
 *            random double numbers
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
 * Function:  Mat_vec_mul(may not be used)
 * Purpose:   Multiply a matrix A by a vector x.  The matrix is distributed
 *            by block rows and the vectors are distributed by blocks
 * In args:   A:  matrix A
 *            x:  vector x
 *            N:  global (and local) number of columns
 * Out args:  y:  The result matrix after multiplication
 * 
 * Errors:    if malloc of local storage on any process fails, all
 *            processes quit.            
 * Notes:     NA 
 */
void Mat_vec_mul(
    double    A[]  /* in  */, 
    double    x[]  /* in  */, 
    double    y[]  /* out */,
    int       N    /* in  */){
        int i, j;
        for (i = 0; i < N; i++) {
            y[i] = 0.0;
            for (j = 0; j < N; j++){
                y[i] += A[i*N+j]*x[j];
            }
        }
}  /* Mat_vec_mul */


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
    double    A[]  /* in  */,
    int       row  /* in  */, 
    double    x[]  /* in  */, 
    int       N    /* in  */){
        if(row >= N || row < 0){
            printf("The row index is invalid(row_idx > N)");
            exit(1);
        }
        int j;
        double res = 0.0; /* out  */
        #pragma opm parallel for reduction(+:res)
        for (j = 0; j < N; j++){
            res += A[row*N+j]*x[j];
        }
    return res;
}  /* row_vec_mul */


/**
 * The code to run as master: send jobs to the workers, and await their replies.
 * There are `worker_count` workers, numbered from 1 up to and including `worker_count`.
 *
 * @param worker_count The number of workers
 * @param startval  The first value to examine.
 * @param nval The number of values to examine.
 * @return The number of values in the specified range.
 */
double *run_as_master(
    int worker_count, 
    int DIMENSION_SIZE, 
    double *VECTOR_RESULT, 
    double *VECTOR_V, 
    double *MATRIX,
    bool last_iteration
) {

    MPI_Bcast(
        VECTOR_V, 
        DIMENSION_SIZE, 
        MPI_DOUBLE,  
        0, 
        MPI_COMM_WORLD
    );
    int active_workers = 0, dimensions_sent = 0;
    int *worker_block_start = (int*) malloc(sizeof(int) * worker_count);
    int *worker_block_sizes = (int*) malloc(sizeof(int) * worker_count);

    int block_size = ceil(DIMENSION_SIZE / worker_count);

    // This variable will store the segment of the matrix to multiply
    double* MATRIX_SEGMENT = (double *) malloc(sizeof(double) * DIMENSION_SIZE * block_size);
        
    for (int worker = 1; worker <= worker_count && dimensions_sent < DIMENSION_SIZE; worker++) {
        int worker_block_size = block_size;
        if ((dimensions_sent + block_size) > DIMENSION_SIZE){
            worker_block_size = DIMENSION_SIZE - dimensions_sent;
        }
        // memcpy( // Get row of matrix to send to worker
        //     MATRIX_SEGMENT, 
        //     MATRIX + dimensions_sent * DIMENSION_SIZE, 
        //     sizeof(double) * DIMENSION_SIZE * worker_block_size
        // );
        // Print_vector("Z", MATRIX_SEGMENT, DIMENSION_SIZE);

        MPI_Send(
            &worker_block_size, 
            1, 
            MPI_INT, worker, 
            TAG_DIMENSION, MPI_COMM_WORLD
        );

        // MPI_Send(
        //     MATRIX_SEGMENT, 
        //     DIMENSION_SIZE * worker_block_size, 
        //     MPI_DOUBLE, worker, 
        //     TAG_MATRIX_BLOCK, MPI_COMM_WORLD
        // );

        worker_block_start[worker] = dimensions_sent;
        worker_block_sizes[worker] = worker_block_size;
        active_workers++;
        dimensions_sent += worker_block_size;
    }

    MPI_Scatter(
        MATRIX,
        block_size * DIMENSION_SIZE,
        MPI_DOUBLE,
        MATRIX_SEGMENT, 0, MPI_DOUBLE, 
        0, 
        MPI_COMM_WORLD
    );

    // printf("Master sent all work...");
    while (active_workers > 0) {
        int worker;
        double * result = (double *) malloc(sizeof(double) * block_size);

        // await_result(&worker, result, worker_block_sizes[worker]);

        MPI_Status status;
        MPI_Recv(
            result, 
            block_size, // last iteration ?? 
            MPI_DOUBLE, MPI_ANY_SOURCE, 
            TAG_RESULT, MPI_COMM_WORLD, &status
        );
        worker = status.MPI_SOURCE;

        memcpy( // Get row of matrix to send to worker
            VECTOR_RESULT + worker_block_start[worker], 
            result, 
            sizeof(double) * worker_block_sizes[worker]
        );

        //turn_off_worker(worker);
        active_workers--;

        // if (dimensions_sent < DIMENSION_SIZE) { // Send next dimension
        //     memcpy( // Get row of matrix to send to worker
        //         MATRIX_SEGMENT, 
        //         MATRIX + dimensions_sent * DIMENSION_SIZE, 
        //         sizeof(double) * DIMENSION_SIZE
        //     );
        //     send_work_command(worker, MATRIX_SEGMENT, DIMENSION_SIZE);
        //     worker_dimension_mapping[worker] = dimensions_sent;
        //     dimensions_sent++;
        // } else { // Turn off worker
        //     if (last_iteration){
        //         turn_off_worker(worker); 
        //     }
        //     active_workers--;
        // }
    }
    return VECTOR_RESULT;
}

/**
 * The code to run as worker: receive jobs, execute them, and terminate when told to.
 */
void run_as_worker(int DIMENSION) {
    printf("wut");
    MPI_Status status;
    int block_size = 0;
    double * VECTOR_V = (double *) malloc (DIMENSION * sizeof(double));

    printf("AJA");
    while (true){
        printf("1");
        MPI_Bcast(
            VECTOR_V, 
            DIMENSION, 
            MPI_DOUBLE,  
            0, 
            MPI_COMM_WORLD
        );
        printf("2");
        MPI_Recv(
            &block_size,
            1,
            MPI_INT, 0, 
            TAG_DIMENSION, MPI_COMM_WORLD,
            &status
        );

        double * MATRIX_BLOCK = (double *) malloc( DIMENSION * block_size * sizeof(double));
        double * result = (double *) malloc (block_size * sizeof(double));

        printf("3");

        MPI_Scatter(
            MATRIX_BLOCK, 1, MPI_DOUBLE,
            MATRIX_BLOCK, 
            DIMENSION * block_size, 
            MPI_DOUBLE, 
            0, 
            MPI_COMM_WORLD
        );

        // MPI_Recv(
        //     MATRIX_BLOCK, 
        //     DIMENSION * block_size, 
        //     MPI_DOUBLE, 0, 
        //     TAG_MATRIX_BLOCK, MPI_COMM_WORLD, 
        //     &status
        // );

        // int size_of_segment;
        // MPI_Get_count(
        //     &status, MPI_DOUBLE, &size_of_segment
        // );
        // if (size_of_segment == 0 || size_of_segment == MPI_UNDEFINED) {
        //     break;  // The master told us to stop.
        // }

        // Perform matrix multiplacion
        #pragma opm parallel for
        for (int i = 0; i < block_size; i ++){
            result[i] = row_vec_mul(MATRIX_BLOCK, i, VECTOR_V, DIMENSION);
        }

        MPI_Send(
            result, 
            block_size, 
            MPI_DOUBLE, 0, 
            TAG_RESULT, 
            MPI_COMM_WORLD
        );

        free(MATRIX_BLOCK);
        free(result);
    }
}

