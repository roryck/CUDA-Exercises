#include <iostream>
#include <string>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _SSE
#include <xmmintrin.h>
#endif

// vector length
#if defined _SSE || defined __ALTIVEC__
#define VECLEN 4
#else
#define VECLEN 1
#endif

// The standard test case sizes
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 16384
#endif

extern "C"
{
#include "Voigt.h"
    // returns a sequence of random floating point signs (+/- 1.0f)
    float ransign(long int *idum);
    int timeval_subtract (struct timeval *result, struct timeval*x, struct timeval*y);
}

using namespace std;

int main(int argc, char* argv[])
{
    // Timers
    struct timeval start;
    struct timeval stop;
    struct timeval result;
    float t_msec_wc;         // wallclock time in msec

    long int SEED = 80305;
    float sign;

    int i=0;
    int j=0;
    int nThds, tId; 
    int nPointsPerThread, nRowPerThread, nWordPerRow;
#ifdef _OPENMP
    nThds = 4;
#else
    nThds = 1;
#endif
    nPointsPerThread = PROBLEM_SIZE*PROBLEM_SIZE/nThds;
    nRowPerThread = PROBLEM_SIZE/nThds;
    nWordPerRow = PROBLEM_SIZE/VECLEN;

    cout << "GPU Grid Decomposition:" << endl;
    cout << "  " << PROBLEM_SIZE << "x" << PROBLEM_SIZE << " total points in grid" << endl;
    cout << "  " << nThds << " Thread(s) " << endl;
    cout << "  " << nRowPerThread << " rows processed per thread" << endl;
#if defined _SSE || defined __ALTIVEC__
    cout << "  " << nWordPerRow << " vectors per row" << endl;
    cout << "  " << VECLEN << " points per vector" << endl;  
#else
    cout << "  " << nWordPerRow << " points per row" << endl;
#endif
    cout << endl;


    float *damp, *offs, *vval;
    int startElem;

    float vals;
    float step = 32.0/PROBLEM_SIZE;

    assert(FLT_EPSILON < step);

#ifdef _SSE
    // align arrays on 16 byte boundaries
    float *damping = (float *) _mm_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float), 16);
    float *offset = (float *) _mm_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float), 16);
    float *voigt = (float *) _mm_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float), 16);
#elif defined __ALTIVEC__
    // align arrays on 16 byte boundaries 
    float *damping = (float *) vec_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
    float *offset = (float *) vec_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
    float *voigt = (float *) vec_malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
#else
    // use the native malloc alilgnment
    float *damping = (float *) malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
    float *offset = (float *) malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
    float *voigt = (float *) malloc(PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float));
#endif

    /* assign some initial data values    */
    /* We want the offsets to be positive */
    vals = 10.0f;
    for(i=0; i<PROBLEM_SIZE; ++i){
        vals += step;
        for(j=0; j<PROBLEM_SIZE; ++j){
            sign = ransign(&SEED);
            damping[i*PROBLEM_SIZE + j] = vals;
            offset[i + j*PROBLEM_SIZE] = vals * sign;
        }
    }
    
    // Get timer for usage so far, notice that we have done nothing
    if (gettimeofday(&start, NULL) != 0)
    {
        cerr << "Error: gettimeofday failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }

#ifdef _OPENMP
    omp_set_num_threads(nThds);
    #pragma omp parallel private(tId, startElem, damp, offs, vval) 
    {
      tId = omp_get_thread_num();
      startElem = tId*nPointsPerThread;
#else
      tId = 0;
      startElem = 0;
#endif

      /*** ----  main compute kernel ----- ***/
      /*** this is where the magic happens ***/
      my_voigt(&damping[startElem], &offset[startElem], &voigt[startElem], PROBLEM_SIZE*PROBLEM_SIZE/nThds);

#ifdef _OPENMP
    }
#endif


    if (gettimeofday(&stop, NULL) != 0)
    {
        cerr<<"getrusage failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }
    timeval_subtract(&result, &stop, &start);
    t_msec_wc = (1000.0f * float(result.tv_sec)) + (float(result.tv_usec) / 1000.0f);

    // print a few values for verification
    cout << endl << "Verification values:"<<endl;
    cout         << "-------------------"<<endl;
    for(i=PROBLEM_SIZE/2; i<PROBLEM_SIZE/2 + 5; i++){
       for(j=0; j<2; j++){
          cout << voigt[i + j*PROBLEM_SIZE] << " ";
       }
       cout << endl;
    }
    cout << "-------------------"<<endl;

    // print timing info
    cout << "  Total wallclock time: "<<t_msec_wc<<" msec"<<endl;

#ifdef _SSE
  _mm_free(damping);
  _mm_free(offset);
  _mm_free(voigt);
#elif defined __ALTIVEC__
  vec_free(damping);
  vec_free(offset);
  vec_free(voigt);
#else
  free(damping);
  free(offset);
  free(voigt);
#endif
    
}
