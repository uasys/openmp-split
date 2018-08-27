/**
 * 2mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#define THREADS 128
#define BLOCKS1 128
#define BLOCKS2 128

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

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

void mm2_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E) {

#pragma omp target data map(to : A[ : NI *NK], B[ : NK *NJ], D[ : NJ *NL]) map(from : E[ : NI *NJ], C[ : NI *NJ])
{
        #pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS1)
        {
	#pragma omp distribute parallel for collapse(2)
	for (int i = 0; i < NI; i++) {
      		for (int j = 0; j < NJ; j++) {
        		C[i * NJ + j] = 0.0;
        		for (int k = 0; k < NK; ++k) {
          			C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        		}
      		}
    	}
        }
        #pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS2)
        {
	#pragma omp distribute parallel for collapse(2)
    	for (int i = 0; i < NI; i++) {
      		for (int j = 0; j < NL; j++) {
        		E[i * NL + j] = 0.0;
        		for (int k = 0; k < NJ; ++k) {
          			E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
        		}
      		}
    	}
        }
}
} 

int main(int argc, char **argv) {
  double t_start, t_end, t_start_GPU1, t_end_GPU1, t_start_GPU2, t_end_GPU2;

  int fail = 0;

  DATA_TYPE *C;
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *D;
  DATA_TYPE *E;

  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));

  fprintf(stdout,"<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D) >>\n");

  t_start_GPU1 = rtclock();

  C = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));

  t_end_GPU1 = rtclock();

  init_array(A, B, C, D);

  t_start_GPU2 = rtclock();

  mm2_OMP(A, B, C, D, E);
  
  free(C);
  free(D);

  t_end_GPU2 = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", (t_end_GPU1 - t_start_GPU1)+(t_end_GPU2-t_start_GPU2));

  free(A);
  free(B);
  free(E);

  return fail;
}
