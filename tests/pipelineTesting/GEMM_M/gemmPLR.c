/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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

#define NI SIZE
#define NJ SIZE
#define NK SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gemm_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {

  cudaHostRegister(A, NI * NK * sizeof(DATA_TYPE),0);
  cudaHostRegister(C, NI * NJ * sizeof(DATA_TYPE),0);

  DATA_TYPE *A1, *A2, *A3, *A4;
  A1 = A;
  A2 = A1 + NI*NK/4;
  A3 = A2 + NI*NK/4;
  A4 = A3 + NI*NK/4;

  DATA_TYPE  *C1, *C2, *C3, *C4;
  C1 = C;
  C2 = C1 + NI*NJ/4;
  C3 = C2 + NI*NJ/4;
  C4 = C3 + NI*NJ/4;

  int a, b, c;

  #pragma omp target data map(alloc : A1[:NI*NK/4], A2[:NI*NK/4], A3[:NI*NK/4], A4[:NI*NK/4], C1[:NI*NJ/4], C2[:NI*NJ/4], C3[:NI*NJ/4], C4[:NI*NJ/4]) \
				map(to : B[:NK*NJ])
  {

  #pragma omp target update to(A1[:NI*NK/4], C1[:NI*NJ/4])
  #pragma omp target update to(A2[:NI*NK/4], C2[:NI*NJ/4]) nowait depend(out: a)
  #pragma omp target teams distribute parallel for thread_limit(THREADS)
  {
  for (int i = 0; i < NI/4; i++) {
    for (int j = 0; j < NJ; j++) {
      C1[i * NJ + j] *= BETA;
      for (int k = 0; k < NK; ++k) {
        C1[i * NJ + j] += ALPHA * A1[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  #pragma omp target update from(C1[:NI*NJ/4]) nowait
  #pragma omp target update to(A3[:NI*NK/4], C3[:NI*NJ/4]) nowait depend(out: b)
  #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: a)
  {
  for (int i = 0; i < NI/4; i++) {
    for (int j = 0; j < NJ; j++) {
      C2[i * NJ + j] *= BETA;
      for (int k = 0; k < NK; ++k) {
        C2[i * NJ + j] += ALPHA * A2[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  #pragma omp target update from(C2[:NI*NJ/4]) nowait
  #pragma omp target update to(A4[:NI*NK/4], C4[:NI*NJ/4]) nowait depend(out: c)
  #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: b)
  {
  for (int i = 0; i < NI/4; i++) {
    for (int j = 0; j < NJ; j++) {
      C3[i * NJ + j] *= BETA;
      for (int k = 0; k < NK; ++k) {
        C3[i * NJ + j] += ALPHA * A3[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  #pragma omp target update from(C3[:NI*NJ/4]) nowait
  #pragma omp target teams distribute parallel for thread_limit(THREADS) depend(in: c)
  {
  for (int i = 0; i < NI/4; i++) {
    for (int j = 0; j < NJ; j++) {
      C4[i * NJ + j] *= BETA;
      for (int k = 0; k < NK; ++k) {
        C4[i * NJ + j] += ALPHA * A4[i * NK + k] * B[k * NJ + j];
      }
    }
  }
  }
  #pragma omp target update from(C4[:NI*NJ/4])
  }
  cudaHostUnregister(A);
  cudaHostUnregister(C);
}

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
    }
  }
}

int main(int argc, char *argv[]) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;

  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix-multiply C=alpha.A.B+beta.C >>\n");

  t_start1 = rtclock();
  
  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
 
  t_end1 = rtclock();

  init(A, B, C);

  t_start2 = rtclock();

  gemm_OMP(A, B, C);

  free(A);
  free(C);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1)+(t_end2-t_start2));

  free(B);

  return fail;
}
