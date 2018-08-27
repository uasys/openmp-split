/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#define THREADS 128

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime_api.h>
#include "BenchmarksUtil.h"

#define N SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {

  DATA_TYPE *A1, *A2, *A3, *A4;
  A1 = A;
  A2 = A1 + N*N/4;
  A3 = A2 + N*N/4;
  A4 = A3 + N*N/4;

  int a, b, c;
  #pragma omp target data map(alloc : A1[:N*N/4], A2[:N*N/4], A3[:N*N/4], A4[:N*N/4]) \
	map(to : B[:N*N], x[:N], tmp[:N]) map(tofrom : y[:N])
  {
	#pragma omp target update to(A1[:N*N/4])
	#pragma omp target update to(A2[:N*N/4]) nowait depend(out: a)
	#pragma omp target teams distribute parallel for thread_limit(THREADS)
  	for (int i = 0; i < N/4; i++) {
    		tmp[i] = 0;
    		y[i] = 0;
    		for (int j = 0; j < N; j++) {
      			tmp[i] = A1[i * N + j] * x[j] + tmp[i];
      			y[i]   = B[i * N + j] * x[j] + y[i];
    		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
  	}
	#pragma omp target update to(A3[:N*N/4]) nowait depend(out: b)
	#pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: a)
  	for (int i = 0; i < N/4; i++) {
    		tmp[i+N/4] = 0;
    		y[i+N/4] = 0;
    		for (int j = 0; j < N; j++) {
      			tmp[i+N/4] = A2[i * N + j] * x[j] + tmp[i+N/4];
      			y[i+N/4]   = B[(i+N/4) * N + j] * x[j] + y[i+N/4];
    		}
		y[i+N/4] = ALPHA * tmp[i+N/4] + BETA * y[i+N/4];
  	}
	#pragma omp target update to(A4[:N*N/4]) nowait depend(out: c)
	#pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: b)
  	for (int i = 0; i < N/4; i++) {
    		tmp[i+N/2] = 0;
    		y[i+N/2] = 0;
    		for (int j = 0; j < N; j++) {
      			tmp[i+N/2] = A3[i * N + j] * x[j] + tmp[i+N/2];
      			y[i+N/2]   = B[(i+N/2) * N + j] * x[j] + y[i+N/2];
    		}
		y[i+N/2] = ALPHA * tmp[i+N/2] + BETA * y[i+N/2];
  	}
	
	#pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: c)
  	for (int i = 0; i < N/4; i++) {
    		tmp[i+N*3/4] = 0;
    		y[i+N*3/4] = 0;
    		for (int j = 0; j < N; j++) {
      			tmp[i+N*3/4] = A4[i * N + j] * x[j] + tmp[i+N*3/4];
      			y[i+N*3/4]   = B[(i+N*3/4) * N + j] * x[j] + y[i+N*3/4];
    		}
		y[i+N*3/4] = ALPHA * tmp[i+N*3/4] + BETA * y[i+N*3/4];
  	}
  }
}

void init(DATA_TYPE *A, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

int main(int argc, char *argv[]) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *tmp;

  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  tmp = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");
  
  t_start1 = rtclock();


  cudaHostAlloc((void**)&A, N * N * sizeof(DATA_TYPE), 0);

  t_end1 = rtclock();

  init(A, x);

  t_start2 = rtclock();

  gesummv_OMP(A, B, x, y, tmp);

  cudaFreeHost(A);
 
  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2-t_start2));

  free(B);
  free(x);
  free(y);
  free(tmp);

  return fail;
}
