/**
 * atax.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#define THREADS 128
#define BLOCKS1 (SIZE+127)/128
#define BLOCKS2 (SIZE+127)/128

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime_api.h>
#include "BenchmarksUtil.h"

#define NX SIZE
#define NY SIZE

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *x, DATA_TYPE *A) {
  int i, j;

  for (i = 0; i < NX; i++) {
    x[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void atax_OMP(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {

  cudaHostRegister(y, NY * sizeof(DATA_TYPE),0);
  cudaHostRegister(tmp, NX * sizeof(DATA_TYPE),0);
  int a;
  for (int i = 0; i < NY; i++) {
    y[i] = 0;	
  }

  #pragma omp target data map(to : A[ : NX *NY], x[ : NY], tmp[:NX]) map(from : y[ : NY])
  {

  #pragma omp target update to(y[:NY]) nowait depend(out: a)
  #pragma omp target teams distribute parallel for num_teams(BLOCKS1) thread_limit(THREADS)
  {
    for (int i = 0; i < NX; i++) {
      tmp[i] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i] = tmp[i] + A[i * NY + j] * x[j];
      }
    }
  }

  #pragma omp target update from(tmp[:NX]) nowait
  #pragma omp target teams distribute parallel for num_teams(BLOCKS2) thread_limit(THREADS) depend(in: a)
  { 
    for (int j = 0; j < NY; j++)
      for (int i = 0; i < NX; i++) {
        y[j] = y[j] + A[i * NY + j] * tmp[i];
      }
  }
  }
  cudaHostUnregister(y);
  cudaHostUnregister(tmp);
}

int main(int argc, char **argv) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *tmp;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Transpose and Vector Multiplication >>\n");

  t_start1 = rtclock();  

  y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  t_end1 = rtclock();

  init_array(x, A);

  t_start2 = rtclock();

  atax_OMP(A, x, y, tmp);

  free(y);
  free(tmp);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2 - t_start2));

  free(A);
  free(x);

  return fail;
}
