/**
 * syrk.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#define THREADS 128
#define BLOCKS1 448
#define BLOCKS2 112

#include "BenchmarksUtil.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

/* Problem size */
#define N SIZE
#define M SIZE

/* Declared constant values for alpha and beta */
/* (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i * M + j] = ((DATA_TYPE)i * j) / N;
    }
    for (j = 0; j < M; j++) {
      D[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
}

void syrkGPU(DATA_TYPE *A, DATA_TYPE *Dinit, DATA_TYPE *D1, DATA_TYPE *D2) {

  #pragma omp target data map(to : A[ : N *M], Dinit[ : N *M]) map(tofrom : D1[ : N *M], D2[ : N *M])
  {
  
  #pragma omp target teams num_teams(BLOCKS1) thread_limit(THREADS)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        D1[i * M + j] = Dinit[i * M + j] * beta;
      }
    }
  }
  
  #pragma omp target teams num_teams(BLOCKS2) thread_limit(THREADS)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        D2[i * N + j] = D1[i * N + j];
        for (int k = 0; k < M; k++) {
          D2[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }
    }
  }
  }
}

int main() {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *Dinit;
  DATA_TYPE *D1;
  DATA_TYPE *D2;

  Dinit = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  D2 = (DATA_TYPE *)calloc(N * M, sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-k operations >>\n");

  t_start1 = rtclock();
  
  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  D1 = (DATA_TYPE *)calloc(N * M, sizeof(DATA_TYPE));

  t_end1 = rtclock();

  init_arrays(A, Dinit);
  
  t_start2 = rtclock();

  syrkGPU(A, Dinit, D1, D2);

  free(A);
  free(D1);
  
  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.61fs\n", (t_end1-t_start1)+(t_end2-t_start2));

  free(D2);
  free(Dinit);
  return fail;
}
