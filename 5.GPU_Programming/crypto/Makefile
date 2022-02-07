CFLAGS = -O2 -Wall
CC = gcc
CUFLAGS = -O2 -g --ptxas-options=-v #-gencode arch=compute_35,code=compute_35
NVCC = nvcc

all: crypto_array

crypto_array: timer.o crypto_array.o
	$(NVCC) $(CUFLAGS) -o crypto_array timer.o crypto_array.o

clean:
	rm -f *.o crypto_array

%.o: %.cc
	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CUFLAGS) -o $@ -c $<
