/**
 * bicg.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
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
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

#define NX SIZE
#define NY SIZE

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r) {
  int i, j;

  for (i = 0; i < NX; i++) {
    r[i] = i * M_PI;
    for (j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * j) / NX;
    }
  }

  for (i = 0; i < NY; i++) {
    p[i] = i * M_PI;
  }
}

void bicg_OMP(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, DATA_TYPE *p,
              DATA_TYPE *q) {
  int i, j;

  for (i = 0; i < NY; i++) {
    s[i] = 0.0;
  }

  #pragma omp target data map(to : A[ : NX *NY], p[ : NY], r[ : NX]) map(tofrom : s[ : NY], q[ : NX])
  {
   
  #pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS1)
  {
    #pragma omp distribute parallel for private(i)
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        s[j] = s[j] + r[i] * A[i * NY + j];
      }
    }
  }

  #pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS2)
  {
    #pragma omp distribute parallel for private(j)
    for (i = 0; i < NX; i++) {
      q[i] = 0.0;
      for (j = 0; j < NY; j++) {
        q[i] = q[i] + A[i * NY + j] * p[j];
      }
    }
  }
  }
}

int main(int argc, char **argv) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *r;
  DATA_TYPE *s;
  DATA_TYPE *p;
  DATA_TYPE *q;

  A = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
  r = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  fprintf(stdout, "<< BiCG Sub Kernel of BiCGStab Linear Solver >>\n");

  t_start1 = rtclock();
  
  p = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  s = (DATA_TYPE *)malloc(NY * sizeof(DATA_TYPE));
  q = (DATA_TYPE *)malloc(NX * sizeof(DATA_TYPE));

  t_end1 = rtclock();
 
  init_array(A, p, r);

  t_start2 = rtclock();

  bicg_OMP(A, r, s, p, q);

  free(p);
  free(s);
  free(q);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1 - t_start1) + (t_end2 - t_start2));

  free(A);
  free(r);

  return fail;
}
