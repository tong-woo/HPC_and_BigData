#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void vectorTransformKernel(double* A, double* B, double* Result) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    Result[i] = Result[i] + A[i]*B[i];
// insert operation here
}

void vectorTransformCuda(int n, double* a, double* b, double* result) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    double* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(double)));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(double)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(double)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return;
    }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(double), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    for(int j = 0; j<5; j++){
    	vectorTransformKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    }
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    cout << "vector-transform (kernel): \t\t" << kernelTime1 << endl;
    cout << "vector-transform (memory): \t\t" << memoryTime  << endl;
}

int vectorTransformSeq(int n, double* a, double* b, double* result){
  int i,j; 

  timer sequentialTime = timer("Sequential");
  
  sequentialTime.start();
  for (j=0; j<5; j++) {
    for (i=0; i<n; i++) {
	result[i] = result[i]+a[i]*b[i];
    }
  }
  sequentialTime.stop();
  
  cout << "vector-transform (sequential): \t\t" << sequentialTime << endl;
  return 0;
}

int main(int argc, char* argv[]) {
    int n[4] = {1024, 65536, 655360, 1000000};
    for(int j = 0; j < sizeof(n)/sizeof(int); j++){
        double* a = new double[n[j]];
        double* b = new double[n[j]];
        double* result = new double[n[j]];
        double* result_s = new double[n[j]];

        if (argc > 1) n[j] = atoi(argv[1]);

        cout << "\nIteratively transform vector A with vector B of " << n[j] << " integer elements." << endl;
        // initialize the vectors.
        for(int i=0; i<n[j]; i++) {
            a[i] = i;
            b[i] = 0.1*i;
            result[i]=0;
            result_s[i]=0;
        }

        vectorTransformSeq(n[j], a, b, result_s);

        vectorTransformCuda(n[j], a, b, result);

        // verify the resuls
        for(int i=0; i<n[j]; i++) {
    //	  if (result[i]!=result_s[i]) {
            if (fabs(result[i] - result_s[i]) >0.001){
                std::cout << "error in results! Element " << i << " is " << result[i] << ", but should be " << result_s[i] << std::endl; 
                exit(1);
            }
        }
        delete[] a;
        delete[] b;
        delete[] result;
	std::cout<<"results OK!" <<std::endl;	
    }
    return 0;
}
