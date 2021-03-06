#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
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


__global__ void encryptKernel(int len, char* deviceDataIn, char* deviceDataOut) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    //int key_len = sizeof(key) / sizeof(int) + 1;

    int key[] ={1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70,
		71,72,73,74,75,76,77,78,79,80,
		81,82,83,84,85,86,87,88,89,90,
		91,92,93,94,95,96,97,98,99,100,
		101,102,103,104,105,106,107,108,109,110,
		111,112,113,114,115,116,117,118,119,120,
		121,122,123,124,125,126,127,128};
    int key_cur = index % len;

    if (deviceDataIn[index] >= 'a' && deviceDataIn[index] <= 'z')
	deviceDataOut[index] = (deviceDataIn[index] - 'a' + key[key_cur]) % 26 + 97;
    else if (deviceDataIn[index] >= 'A' && deviceDataIn[index] <= 'Z')
	deviceDataOut[index] = (deviceDataIn[index] - 'A' + key[key_cur]) % 26 + 65;
    else
	deviceDataOut[index] = deviceDataIn[index];
}

__global__ void decryptKernel(int len, char* deviceDataIn, char* deviceDataOut) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;    
    //int key_len = sizeof(key) / sizeof(int) + 1;
    int key_cur = index % len;
    
    int key[] ={1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70,
		71,72,73,74,75,76,77,78,79,80,
		81,82,83,84,85,86,87,88,89,90,
		91,92,93,94,95,96,97,98,99,100,
		101,102,103,104,105,106,107,108,109,110,
		111,112,113,114,115,116,117,118,119,120,
		121,122,123,124,125,126,127,128};
    if (deviceDataIn[index] >= 'a' && deviceDataIn[index] <= 'z')
	deviceDataOut[index] = (deviceDataIn[index] - 'a' - (key[key_cur]%26) + 26) % 26 + 97;
    else if (deviceDataIn[index] >= 'A' && deviceDataIn[index] <= 'Z')
	deviceDataOut[index] = (deviceDataIn[index] - 'A' - (key[key_cur]%26) + 26) % 26 + 65;
    else
	deviceDataOut[index] = deviceDataIn[index];
}

int fileSize() {
  int size; 

  ifstream file ("original.data", ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.close();
  }
  else {
    cout << "Unable to open file";
    size = -1; 
  }
  return size; 
}

int readData(char *fileName, char *data) {

  streampos size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.seekg (0, ios::beg);
    file.read (data, size);
    file.close();

    cout << "The entire file content is in memory." << endl;
  }
  else cout << "Unable to open file" << endl;
  return 0;
}

int writeData(int size, char *fileName, char *data) {
  ofstream file (fileName, ios::out|ios::binary|ios::trunc);
  if (file.is_open())
  {
    file.write (data, size);
    file.close();

    cout << "The entire file content was written to file." << endl;
    return 0;
  }
  else cout << "Unable to open file";

  return -1; 
}

int EncryptSeq (int n, int key[], int len, char* data_in, char* data_out) 
{  
  int i;
  //int key_len = sizeof(key) / sizeof(int);
  int key_cur; //current offset
  timer sequentialTime = timer("Sequential encryption");
  
  sequentialTime.start();
  /*
  //single key
  for (i=0; i<n; i++)
  {
    if (data_in[i] >= 'a' && data_in[i] <= 'z')
	data_out[i] = (data_in[i] - 'a' + key) % 26 + 97;
    else if (data_in[i] >= 'A' && data_in[i] <= 'Z')
	data_out[i] = (data_in[i] - 'A' + key) % 26 + 65;
    else
        data_out[i] = data_in[i];
 
  }
  */
  
  //larger key
  for (i=0; i<n; i++)
  {
    key_cur = i % len; 
    if (data_in[i] >= 'a' && data_in[i] <= 'z')
	data_out[i] = (data_in[i] - 'a' + key[key_cur]) % 26 + 97;
    else if (data_in[i] >= 'A' && data_in[i] <= 'Z')
	data_out[i] = (data_in[i] - 'A' + key[key_cur]) % 26 + 65;
    else
        data_out[i] = data_in[i];
  }
  
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;
  
  return 0; 
}

int DecryptSeq (int n, int key[], int len, char* data_in, char* data_out)
{
  int i;
  //int key_len = sizeof(key) / sizeof(int);
  int key_cur;

  timer sequentialTime = timer("Sequential decryption");

  sequentialTime.start();
  /*
  //single key
  for (i=0; i<n; i++) 
  { 
    if (data_in[i] >= 'a' && data_in[i] <= 'z')
	data_out[i] = (data_in[i] - 'a' - key + 26) % 26 + 97;
    else if (data_in[i] >= 'A' && data_in[i] <= 'Z')
	data_out[i] = (data_in[i] - 'A' - key + 26) % 26 + 65;
    else 
        data_out[i] = data_in[i];
  }
  */
  
  //larger key
  for (i=0; i<n; i++ )
  {
    key_cur = i % len; 
    if (data_in[i] >= 'a' && data_in[i] <= 'z')
	data_out[i] = (data_in[i] - 'a' - (key[key_cur]%26) + 26 ) % 26 + 97;
    else if (data_in[i] >= 'A' && data_in[i] <= 'Z')
	data_out[i] = (data_in[i] - 'A' - (key[key_cur]%26) + 26) % 26 + 65;
    else
        data_out[i] = data_in[i];
  }
  
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

  return 0;
}


int EncryptCuda (int n, int len, char* data_in, char* data_out) {
    int threadBlockSize = 512;
    //int key_len = sizeof(key) / sizeof(int);
    //int key_cur;
    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
   
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    encryptKernel<<<(n/threadBlockSize)+1, threadBlockSize>>>( len, deviceDataIn, deviceDataOut);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int DecryptCuda (int n, int len, char* data_in, char* data_out) {
    int threadBlockSize = 512;

    //int key_len = sizeof(key) / sizeof(int);
    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    decryptKernel<<<(n/threadBlockSize)+1, threadBlockSize>>>(len, deviceDataIn, deviceDataOut);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int main(int argc, char* argv[]) {
    int n;
    int key[] ={1,2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70,
		71,72,73,74,75,76,77,78,79,80,
		81,82,83,84,85,86,87,88,89,90,
		91,92,93,94,95,96,97,98,99,100,
		101,102,103,104,105,106,107,108,109,110,
		111,112,113,114,115,116,117,118,119,120,
		121,122,123,124,125,126,127,128};
    int key_len = sizeof(key) / sizeof(int);
    printf("The length of key:%d\n",key_len);
    n = fileSize();

    if (n == -1) {
	cout << "File not found! Exiting ... " << endl; 
	exit(0);
    }

    char* data_in = new char[n];
    char* data_out = new char[n];    
    readData("original.data", data_in); 

    cout << "Encrypting a file of " << n << " characters." << endl;

    EncryptSeq(n, key, key_len, data_in, data_out);
    writeData(n, "sequential.data", data_out);
    EncryptCuda(n, key_len, data_in, data_out);
    writeData(n, "cuda.data", data_out);  

    readData("cuda.data", data_in);

    cout << "Decrypting a file of " << n << " characters" << endl;
    DecryptSeq(n, key, key_len, data_in, data_out);
    writeData(n, "sequential_decrypted.data", data_out);
    DecryptCuda(n, key_len, data_in, data_out); 
    writeData(n, "recovered.data", data_out); 
 
    delete[] data_in;
    delete[] data_out;
    
    return 0;
}
