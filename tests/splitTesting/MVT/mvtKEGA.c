/**
 * mvt.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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

#define N SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
                DATA_TYPE *y2) {
  int i, j;

  for (i = 0; i < N; i++) {
    x1[i] = ((DATA_TYPE)i) / N;
    x2[i] = ((DATA_TYPE)i + 1) / N;
    y1[i] = ((DATA_TYPE)i + 3) / N;
    y2[i] = ((DATA_TYPE)i + 4) / N;
    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void runMvt_OMP(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *x2, DATA_TYPE *y1,
                DATA_TYPE *y2) {
  cudaHostRegister(x2, N * sizeof(DATA_TYPE),0);
  cudaHostRegister(y2, N * sizeof(DATA_TYPE),0);
  cudaHostRegister(x1, N * sizeof(DATA_TYPE),0);

  int i, j;
  int b;
  #pragma omp target data map(to: a[:N*N], x1[:N], y1[:N]) map(alloc : y2[:N])  map(from: x2[:N])
  {
 
  #pragma omp target update to(x2[:N],y2[:N]) nowait depend(out: b)
  #pragma omp target teams distribute parallel for private(j) num_teams(BLOCKS1) thread_limit(THREADS)
  {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        x1[i] = x1[i] + a[i * N + j] * y1[j];
      }
    }
  }

  #pragma omp target update from(x1[:N]) nowait
  #pragma omp target teams distribute parallel for private(j) num_teams(BLOCKS2) thread_limit(THREADS) depend(in: b)
  {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        x2[i] = x2[i] + a[j * N + i] * y2[j];
      }
    }
  }
  }
  cudaHostUnregister(x2);
  cudaHostUnregister(y2);
  cudaHostUnregister(x1);
}

int main() {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *a;
  DATA_TYPE *x1;
  DATA_TYPE *x2;
  DATA_TYPE *y_1;
  DATA_TYPE *y_2;

  a = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Matrix Vector Product and Transpose >>\n");

  t_start1 = rtclock();

  y_2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x1 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  x2 = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  t_end1 = rtclock();

  init_array(a, x1, x2, y_1, y_2);

  t_start2 = rtclock();

  runMvt_OMP(a, x1, x2, y_1, y_2);

  free(y_2);
  free(x1);
  free(x2);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2 - t_start2));

  free(a);
  free(y_1);

  return fail;
}
