

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
 * Notes:    1. Add an overall error handling function in the future if we have free time
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

/*--------------------------------------------------------------------*/
int main(void) {
   double* A;
   double* x;
   double* y;
   int n;
   double result;
   double start, end;
   int R = 100;
   int N = 10000;
//    Set_dims(&n, &N);                 //Hover on the functions to see details
   Allocate_dynamic_arrays(&A, &x, &y, N);
    //set seed to generate random nums
   Build_matrix(A, N);               //Matrix array stored in A
//    Print_matrix("A", A, N);  
   Build_vector(x, N);               //Vector array stored in x
//    Print_vector("x", x, N);
   /*uncomment below to verify if the first element of y vector is the same as result value*/
   start = omp_get_wtime();
   for(int i=0;i < R;i++){
       Mat_vec_mul(A, x, y, N);
       x = y;
   }
   end  = omp_get_wtime();
   printf("Time estimate: %.4f", end-start);

//    Print_vector("y", y, N);
   // result = row_vec_mul(A, 0, x, N); // Now the product of vectors(A[0] and X) has been stored in result
//    printf("\n%f ", result);
   free(A);
   free(x);
   free(y);
   return 0;
}  /* main */


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
 *             ud=sing dynamic arrays
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

        *A_pp = (double*) malloc(N*N*sizeof(double));
        *x_pp = (double*) malloc(N*sizeof(double));
        *y_pp = (double*) malloc(N*sizeof(double));
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
         A[i*N + j] = ((double) rand());
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
      x[i] = ((double) rand());
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
        printf("\n");

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
        #pragma omp parallel for
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
 *            row: the row index of matrix (e.g. 0,1,2...N)
 *            x:   vector x
 *            N:   number of columns
 * out args:  res: stores the duble value of vector product
 * 
 * Errors:    if malloc of local storage on any process fails, all
 *            processes quit.            
 * Notes:     NA
 */
double row_vec_mul(
    double    A[]  /* in  */,
    int       row  /* in  */, 
    double    x[]  /* in  */, 
    int       N    /* in  */){
        int j;
        double res = 0.0;
        for (j = 0; j < N; j++){
            res += A[row*N+j]*x[j];
        }
    return res;
}  /* row_vec_mul */
