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

#include "BenchmarksUtil.h"

#define N SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y,
                 DATA_TYPE *tmp) {
  #pragma omp target teams distribute parallel for map(to : A[ : N *N], B[ : N *N], x[ : N], tmp[ : N]) map(tofrom : y[ : N]) thread_limit(THREADS)
  for (int i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (int j = 0; j < N; j++) {
      tmp[i] = A[i * N + j] * x[j] + tmp[i];
      y[i] = B[i * N + j] * x[j] + y[i];
    }

    y[i] = ALPHA * tmp[i] + BETA * y[i];
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
 
  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));

  t_end1 = rtclock();

  init(A, x);

  t_start2 = rtclock();

  gesummv_OMP(A, B, x, y, tmp);

  free(A);
    
  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2-t_start2));

  free(B);
  free(x);
  free(y);
  free(tmp);

  return fail;
}
