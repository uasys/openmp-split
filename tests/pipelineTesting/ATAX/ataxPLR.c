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

  cudaHostRegister(A, NX * NY * sizeof(DATA_TYPE), 0);

  for (int i = 0; i < NY; i++) {
    y[i] = 0;
  }
 
  DATA_TYPE *A1, *A2, *A3, *A4;
  A1 = A;
  A2 = A1 + NX*NX/4;
  A3 = A2 + NX*NX/4;
  A4 = A3 + NX*NX/4;

  int a, b, c;
  #pragma omp target data map(alloc : A1[:NX*NY/4],A2[:NX*NY/4],A3[:NX*NY/4],A4[:NX*NY/4]) map(to : x[ : NY]) map(tofrom : tmp[ : NX], y[ : NY])
  {
    #pragma omp target update to(A1[:NX*NY/4])
    #pragma omp target update to(A2[:NX*NY/4]) nowait depend(out:a )
    #pragma omp target teams distribute parallel for thread_limit(THREADS)
    for (int i = 0; i < NX/4; i++) {
      tmp[i] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i] = tmp[i] + A1[i * NY + j] * x[j];
      }
    }
    #pragma omp target update to(A3[:NX*NY/4]) nowait depend(out: b)
    #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: a)
    for (int i = 0; i < NX/4; i++) {
      tmp[i+NX/4] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i+NX/4] = tmp[i+NX/4] + A2[i * NY + j] * x[j];
      }
    }
    #pragma omp target update to(A4[:NX*NY/4]) nowait depend(out: c)
    #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: b)
    for (int i = 0; i < NX/4; i++) {
      tmp[i+NX/2] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i+NX/2] = tmp[i+NX/2] + A3[i * NY + j] * x[j];
      }
    }

    #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: c)
    for (int i = 0; i < NX/4; i++) {
      tmp[i+NX*3/4] = 0;
      for (int j = 0; j < NY; j++) {
        tmp[i+NX*3/4] = tmp[i+NX*3/4] + A4[i * NY + j] * x[j];
      }
    }

    // Note that the Loop has been reversed
    #pragma omp target teams distribute parallel for thread_limit(THREADS)
    for (int j = 0; j < NY; j++)
      for (int i = 0; i < NX; i++) {
	if (i < NX/4)
	        y[j] = y[j] + A1[i * NY + j] * tmp[i];
	else if (i < NX/2)
	        y[j] = y[j] + A2[(i-NX/4) * NY + j] * tmp[i];
	else if (i < NX*3/4)
	        y[j] = y[j] + A3[(i-NX/2) * NY + j] * tmp[i];
	else
	        y[j] = y[j] + A4[(i-NX*3/4) * NY + j] * tmp[i];
      }
  }

  cudaHostUnregister(A);

}

int main(int argc, char **argv) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *tmp;

  x = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Transpose and Vector Multiplication >>\n");

  t_start1 = rtclock();  

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
 
  t_end1 = rtclock();

  init_array(x, A);

  t_start2 = rtclock();
  
  atax_OMP(A, x, y, tmp);

  free(A);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2 - t_start2));

  free(x);
  free(y);
  free(tmp);

  return fail;
}
