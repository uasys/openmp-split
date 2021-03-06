/**
 * 3mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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

#include "BenchmarksUtil.h"

#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE
#define NM SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NJ; i++) {
    for (j = 0; j < NM; j++) {
      C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
    }
  }

  for (i = 0; i < NM; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void mm3_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G) {

/* E := A*B */
#pragma omp target teams map(to : A[ : NI *NK], B[ : NK *NJ], C[ : NJ *NM], D[ : NM *NL]) map(from : E[ : NI *NJ], F[ : NJ *NL], G[ : NI *NL]) thread_limit(THREADS)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NJ; j++) {
        E[i * NJ + j] = 0;
        for (int k = 0; k < NK; ++k) {
          E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
      }
    }

    /* F := C*D */
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NJ; i++) {
      for (int j = 0; j < NL; j++) {
        F[i * NL + j] = 0;
        for (int k = 0; k < NM; ++k) {
          F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
        }
      }
    }

    /* G := E*F */
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NL; j++) {
        G[i * NL + j] = 0;
        for (int k = 0; k < NJ; ++k) {
          G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  double t_start1, t_end1, t_start2, t_end2;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *F;
  DATA_TYPE *G;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  G = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));

  fprintf(
      stdout,
      "<< Linear Algebra: 3 Matrix Multiplications (E=A.B; F=C.D; G=E.F) >>\n");

  t_start1 = rtclock();

  E = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  F = (DATA_TYPE *)calloc(NJ * NL, sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(NJ * NM * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NM * NL * sizeof(DATA_TYPE));

  t_end1 = rtclock();

  init_array(A, B, C, D);

  t_start2 = rtclock();

  mm3_OMP(A, B, C, D, E, F, G);

  free(E);
  free(F);
  free(C);
  free(D);

  t_end2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end1-t_start1) + (t_end2-t_start2));

  free(A);
  free(B);
  free(G);

  return fail;
}
