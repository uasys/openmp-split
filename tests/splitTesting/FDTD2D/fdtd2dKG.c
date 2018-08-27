/**
 * fdtd2d.c: This file was adapted from PolyBench/GPU 1.0 test suite
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
#define BLOCKS2 448
#define BLOCKS3 448
#define BLOCKS4 448

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

#define tmax 500
#define NX SIZE
#define NY SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey,
                 DATA_TYPE *hz) {
  int i, j;

  for (i = 0; i < tmax; i++) {
    _fict_[i] = (DATA_TYPE)i;
  }

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      ex[i * NY + j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
      ey[i * NY + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
      hz[i * NY + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
    }
  }
}

void runFdtd_OMP(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey,
                 DATA_TYPE *hz) {
  int t, i, j;

#pragma omp target data map(to : _fict_[ : tmax], ex[ : (NX * (NY + 1))], ey[ : ((NX + 1) * NY)]) map(tofrom : hz[ : NX * NY])
{
	for (t = 0; t < tmax; t++) {

		#pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS1)
		{
			#pragma omp distribute parallel for
      			for (j = 0; j < NY; j++) {
        			ey[0 * NY + j] = _fict_[t];
      			}	
		}

		#pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS2)
		{
			#pragma omp distribute parallel for collapse(2)
      			for (i = 1; i < NX; i++) {
        			for (j = 0; j < NY; j++) {
          				ey[i * NY + j] = ey[i * NY + j] - 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
	        		}
	      		}
		}

		#pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS3)
		{
			#pragma omp distribute parallel for collapse(2)
			for (i = 0; i < NX; i++) {
				for (j = 1; j < NY; j++) {
					ex[i * (NY + 1) + j] = ex[i * (NY + 1) + j] - 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
      		 	 	}
     		 	}
		}
		
		#pragma omp target teams thread_limit(THREADS) num_teams(BLOCKS4)
		{
			#pragma omp distribute parallel for collapse(2)
			for (i = 0; i < NX; i++) {
       			 	for (j = 0; j < NY; j++) {
       		   			hz[i * NY + j] = hz[i * NY + j] - 0.7 * (ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
       		 		}
      			}
		}
    	}
}
}

int main() {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *_fict_;
  DATA_TYPE *ex;
  DATA_TYPE *ey;
  DATA_TYPE *hz;

  _fict_ = (DATA_TYPE *)malloc(tmax * sizeof(DATA_TYPE));
  ex = (DATA_TYPE *)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
  ey = (DATA_TYPE *)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
  hz = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));

  fprintf(stdout, "<< 2-D Finite Different Time Domain Kernel >>\n");

  init_arrays(_fict_, ex, ey, hz);

  t_start = rtclock();

  runFdtd_OMP(_fict_, ex, ey, hz);

  t_end = rtclock();

  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);

  return fail;
}
