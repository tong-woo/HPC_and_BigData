
//For the 2nd question in assignment
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  //initialize variables
  int i;
  double pi = 0;
  int niters[7] = {31250000, 62500000, 125000000, 250000000, 500000000, 1000000000, 2000000000};
  int n = 0;
  // Get timing
  double start,end;
  
  // Calculate PI using Leibnitz sum
  /* Fork a team of threads */
  for(n=0; n<7; n++){
    start=omp_get_wtime();
    #pragma omp parallel for reduction(+ : pi)
    for(i = 0; i < niters[n]; i++)
    {
      pi = pi + pow(-1, i) * (4 / (2*((double) i)+1));
    } /* Reduction operation is done. All threads join master thread and disband */

    // Stop timing
    end=omp_get_wtime();

    // Print result
    printf("Pi estimate: %.20f, obtained in %f seconds\n", pi, end-start);
  }
}
