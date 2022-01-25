EXECUTABLES=mat_vec_mul

EXPENSIVE_JUNK += $(EXECUTABLES)

SRC = mat_vec_mul.c

JUNK +=

CFLAGS += -O3 -Wall -W --std=c11 -lm -Wno-cast-function-type
# CXXFLAGS += -O3 -Wall -W --std=c++11 -lm -Wno-cast-function-type
OMP_CFLAGS = $(CFLAGS) -fopenmp
MPI_CFLAGS = $(CFLAGS) -lmpi

help:
	@echo "help\tShow this help text"
	@echo "all\tMake all executables"
	@echo "clean\tThrow away all files that are easy to produce again"
	@echo "empty\tThrow away all files that can be produced again"

all: $(EXECUTABLES)

clean:
	rm -rf $(JUNK)

empty:
	rm -rf $(JUNK) $(EXPENSIVE_JUNK)

mat_vec_mul: mat_vec_mul.c
	mpiCC $(MPI_CFLAGS) -o mat_vec_mul mat_vec_mul.c
