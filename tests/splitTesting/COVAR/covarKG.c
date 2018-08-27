/**
 * covariance.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#define THREADS 128
#define BLOCKS1 (SIZE+127)/128
#define BLOCKS2 448
#define BLOCKS3 (SIZE+127)/128

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

#define M SIZE
#define N SIZE

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *data) {
  int i, j;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
    }
  }
}

void covariance_OMP(DATA_TYPE *data, DATA_TYPE *data2, DATA_TYPE *symmat,
                    DATA_TYPE *mean) {

/* Determine mean of column vectors of input data matrix */

  #pragma omp target  data map(to: data[:(M+1)*(N+1)], data2[:(M+1)*(N+1)]) map(alloc: mean[:(M+1)]) map(tofrom: symmat[:(M+1)*(N+1)])
  {
    #pragma omp target teams num_teams(BLOCKS1) thread_limit(THREADS)
    {
    #pragma omp distribute parallel for
    for (int j = 1; j < (M + 1); j++) {
      mean[j] = 0.0;
      for (int i = 1; i < (N + 1); i++) {
        mean[j] += data[i * (M + 1) + j];
      }
      mean[j] /= FLOAT_N;
    }
    }

    /* Center the column vectors. */
    #pragma omp target teams num_teams(BLOCKS2) thread_limit(THREADS)
    {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 1; i < (N + 1); i++) {
      for (int j = 1; j < (M + 1); j++) {
        data2[i * (M + 1) + j] = data[i * (M + 1) + j] - mean[j];
      }
    }
    }

    /* Calculate the m * m covariance matrix. */
    #pragma omp target teams num_teams(BLOCKS3) thread_limit(THREADS)
    {
    #pragma omp distribute parallel for
    for (int j1 = 1; j1 < (M + 1); j1++) {
      for (int j2 = j1; j2 < (M + 1); j2++) {
        symmat[j1 * (M + 1) + j2] = 0.0;
        for (int i = 1; i < N + 1; i++) {
          symmat[j1 * (M + 1) + j2] +=
            data2[i * (M + 1) + j1] * data2[i * (M + 1) + j2];
        }
        symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
      }
    }
    }
  }
}

int main() {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *data;
  DATA_TYPE *data2;
  DATA_TYPE *symmat;
  DATA_TYPE *mean;

  data = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  data2 = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)calloc((M + 1), sizeof(DATA_TYPE));

  fprintf(stdout, "<< Covariance Computation >>\n");

  init_arrays(data);

  t_start = rtclock();

  symmat = (DATA_TYPE *)calloc((M + 1) * (M + 1), sizeof(DATA_TYPE));

  covariance_OMP(data, data2, symmat, mean);

  free(symmat);

  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  free(data);
  free(data2);
  free(mean);

  return fail;
}
