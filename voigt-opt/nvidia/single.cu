#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include "cutil.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "voigt.h"

// The standard test case sizes
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 4096
#endif

using namespace std;

float ransign(long int *idum);

int main(int argc, char* argv[])
{
    /* check for CUDA devices on host system *
     * and print device info if found        */
    int ndev;
    cudaGetDeviceCount(&ndev);
    struct cudaDeviceProp deviceProp;
    if (ndev == 0)
       printf("No CUDA devices found.\n");
    else if(ndev == 1)
       printf("\nFound %d CUDA device:\n", ndev);
    else
       printf("\nFound %d CUDA devices:\n", ndev);
    for(int dev=0; dev<ndev; ++dev){
       cudaGetDeviceProperties(&deviceProp, dev);
       printf("  CUDA Device %d - %s\n", dev, deviceProp.name);
       printf("    Clock rate:               %10.2f MHz\n", (float)deviceProp.clockRate/1000.0);
       printf("    Global memory:            %10d MB\n", deviceProp.totalGlobalMem/1048576);
       printf("    Constant memory:          %10d KB\n", deviceProp.totalConstMem/1024);
       printf("    Shared memory per block:  %10d KB\n", deviceProp.sharedMemPerBlock/1024);
       printf("    Registers per block:      %10d   \n", deviceProp.regsPerBlock);
       printf("    Warp Size:                %10d \n", deviceProp.warpSize);
       printf("    Max Threads per block:    %10d   \n", deviceProp.maxThreadsPerBlock);
       printf("    Max Block Dims (X  Y  Z): %10d  %5d  %5d\n", deviceProp.maxThreadsDim[0],
                                                                deviceProp.maxThreadsDim[1],
                                                                deviceProp.maxThreadsDim[2]);
       printf("    Max Grid Dims  (X  Y  Z): %10d  %5d  %5d\n", deviceProp.maxGridSize[0],
                                                                deviceProp.maxGridSize[1],
                                                                deviceProp.maxGridSize[2]);
       printf("\n");
    }

    /* configure the problem decomposition */
#ifdef _OPENMP
    int nOmpThds   = 2;                        // number of OpenMP threads to use (one data slice / thread)
#else
    int nOmpThds   = 1;                        // set to 1 if not compiling with OpenMP
#endif
    int nChunk     = 1;                        // number of chunks to break each slice into
    int nStream    = 1;                        // number of streams to process concurrently for each chunk
    int nRowOmp    = PROBLEM_SIZE/nOmpThds;    // number of rows in each OMP thread's slice of the data
    int nRowChunk  = nRowOmp/nChunk;           // number of rows in each chunk
    int nRowStream = nRowChunk/nStream;        // number of rows per stream
    int nthds_x = 64;                          // threads per thread-block in x
    int nthds_y = 2;                           // threads per thread-block in y
    int ntblks_x = PROBLEM_SIZE/nthds_x;       // thread-blocks in grid in x;
    int ntblks_y = nRowStream/nthds_y;         // thread-blocks in grid in y;
    dim3 dimGrid(ntblks_x, ntblks_y);          // grid dimensions
    dim3 dimBlock(nthds_x, nthds_y);           // thread-block dimensions

    /* print info about decomposition */
    cout << "GPU Grid Decomposition:"                                        << endl;
    cout << "  " << PROBLEM_SIZE*PROBLEM_SIZE << " total points"             << endl;
    cout << "  " << nOmpThds << " CPU Threads "                              << endl;
    cout << "  " << nChunk << " chunks of rows per CPU thread"               << endl;
    cout << "  " << nStream << " streams per chunk"                          << endl;
    cout << "  " << ntblks_x << "x" << ntblks_y << " thread-blocks in grid"  << endl;
    cout << "  " << nthds_x << "x" << nthds_y << " threads per thread-block" << endl;
    cout << "  " << PROBLEM_SIZE << " total rows in problem"                 << endl;
    cout << "  " << nRowOmp << " rows per CPU Thread"                        << endl;
    cout << "  " << nRowChunk << " rows per chunk"                           << endl;
    cout << "  " << nRowStream << " rows per stream"                         << endl;
    cout                                                                     << endl;

    /* check that the decompostion works on the hardware */
    if(nthds_x > deviceProp.maxThreadsDim[0])
       cout << "Warning: nthds_x exceeds allowed max" << endl;
    if(nthds_y > deviceProp.maxThreadsDim[1])
       cout << "Warning: nthds_y exceeds allowed max" << endl;
    if(nthds_y*nthds_x > deviceProp.maxThreadsPerBlock)
       cout << "Warning: threads per block exceeds allowed max" << endl;
    if(ntblks_x > deviceProp.maxGridSize[0])
       cout << "Warning: ntblks_x exceeds allowed max" << endl;
    if(ntblks_y > deviceProp.maxGridSize[1])
       cout << "Warning: ntblks_y exceeds allowed max" << endl;

    /* temporaries used for initialization */
    float vals;
    float step = 32.0/PROBLEM_SIZE;
    assert(FLT_EPSILON < step);

    /* this array is used to assign initial values to the thread-local arrays*/
    assert(FLT_EPSILON < step);
    long int SEED = 80305;
    vals = 10.0;
    float v[PROBLEM_SIZE];
    for(int i=0; i<PROBLEM_SIZE; ++i){
       vals += step;
       v[i] = vals;
    }

    /* events and timing variables */
    cudaEvent_t tot_b, tot_e;
    float tot_elapsed;
#ifdef _TIMING_PROF_
    cudaEvent_t ker_b, ker_e, mem_i_b, mem_i_e, mem_o_b, mem_o_e;
    float temp, ker_elapsed,  mem_i_elapsed, mem_o_elapsed; 
#endif
    int startElemOmp;
    int startElemChunk;
    int Offset = nRowChunk*PROBLEM_SIZE / nStream;
    size_t memSize = PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float);
    size_t memSizeOmp = memSize / nOmpThds;
    size_t memSizeChunk = memSizeOmp / nChunk;
    size_t memSizeStream = memSizeChunk / nStream;
    int ompTid, devId;
    int a1, a2;
    cudaStream_t *s;
    int *zerobuffer;
    float *h_d, *h_o, *h_v;
    float *d_damp, *d_offs, *d_vval;
    
    /* array to hold output of computation */
    float *h_vval;
    
#ifdef _OPENMP
    omp_set_num_threads(nOmpThds);
#ifdef _TIMING_PROF_
#pragma omp parallel private(tot_b, tot_e, ker_b, ker_e, mem_i_b, mem_i_e, mem_o_b, mem_o_e, temp,    \
                             tot_elapsed, ker_elapsed, mem_i_elapsed, mem_o_elapsed,                  \
                             ompTid, devId, s, startElemOmp, startElemChunk, a1, a2, zerobuffer, \
                             h_d, h_o, h_v, d_damp, d_offs, d_vval)
#else
#pragma omp parallel private(tot_b, tot_e, tot_elapsed, ompTid, devId, s, \
                             startElemOmp, startElemChunk, a1, a2, zerobuffer, \
                             h_d, h_o, h_v, d_damp, d_offs, d_vval)
#endif //_TIMING_PROF_
    {
      ompTid = omp_get_thread_num();
#else 
      ompTid = 0;
#endif //_OPENMP
      devId = ompTid;
      tot_elapsed = 0.0;
#ifdef _TIMING_PROF_
      ker_elapsed = 0.0;
      mem_i_elapsed = 0.0;
      mem_o_elapsed = 0.0;
#endif
#pragma omp critical
      {
         cout << "CPU Thread " << ompTid << " using device " << devId << endl;
      }
      CUDA_SAFE_CALL( cudaSetDevice(devId) );

      /* create timing events */
      CUDA_SAFE_CALL( cudaEventCreate(&tot_b) );
      CUDA_SAFE_CALL( cudaEventCreate(&tot_e) );
#ifdef _TIMING_PROF_
      CUDA_SAFE_CALL( cudaEventCreate(&ker_b) );
      CUDA_SAFE_CALL( cudaEventCreate(&ker_e) );
      CUDA_SAFE_CALL( cudaEventCreate(&mem_i_b) );
      CUDA_SAFE_CALL( cudaEventCreate(&mem_i_e) );
      CUDA_SAFE_CALL( cudaEventCreate(&mem_o_b) );
      CUDA_SAFE_CALL( cudaEventCreate(&mem_o_e) );
#endif

      /* zero a portion of device memory */
      CUDA_SAFE_CALL( cudaMalloc((void**)&zerobuffer, (int)(.5*deviceProp.totalGlobalMem)) );
      CUDA_SAFE_CALL( cudaMemset((void*)zerobuffer, 0, (int)(.5*deviceProp.totalGlobalMem)) );
      CUDA_SAFE_CALL( cudaFree(zerobuffer) );

      /* allocate space on host and device for input and output data */
      CUDA_SAFE_CALL( cudaMallocHost((void**)&h_d, memSizeOmp) ); 
      CUDA_SAFE_CALL( cudaMallocHost((void**)&h_o, memSizeOmp) );
      CUDA_SAFE_CALL( cudaMallocHost((void**)&h_v, memSizeOmp) );

      /* assign initial values */
      startElemOmp = ompTid*nRowOmp*PROBLEM_SIZE;
#pragma omp critical
      {
        for(int i=0; i<nRowOmp*PROBLEM_SIZE; ++i){
           h_d[i] = v[(i+startElemOmp)/PROBLEM_SIZE];
           h_o[i] = v[(i+startElemOmp)%PROBLEM_SIZE] * ransign(&SEED);
        }
      }

      /* allocate space on device for input / output data */
      CUDA_SAFE_CALL( cudaMalloc((void**)&d_damp, memSizeChunk) );
      CUDA_SAFE_CALL( cudaMalloc((void**)&d_offs, memSizeChunk) );
      CUDA_SAFE_CALL( cudaMalloc((void**)&d_vval, memSizeChunk) );

      /* create nStream streams */
      //CUDA_SAFE_CALL( cudaStreamCreate(&s1) );
      //CUDA_SAFE_CALL( cudaStreamCreate(&s2) );
      s = (cudaStream_t *) malloc(nStream * sizeof(cudaStream_t));
      for(int is=0; is<nStream; ++is)
        CUDA_SAFE_CALL( cudaStreamCreate(&s[is]) );

      /* start timing */
#pragma omp barrier
      CUDA_SAFE_CALL( cudaEventRecord(tot_b,0) );
      for(int iChunk=0; iChunk < nChunk; ++iChunk){

#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(mem_i_b, 0) );
#endif
        /* loop over streams */
        startElemChunk = iChunk*nRowChunk*PROBLEM_SIZE;
        for(int is=0; is<nStream; ++is){
          /* compute starting element of each stream in this chunk */
          a1 = is*Offset;
          a2 = startElemChunk + a1;

          /* transfer data CPU -> GPU */
          CUDA_SAFE_CALL( cudaMemcpyAsync((void*) &d_damp[a1], (void*) &h_d[a2], memSizeStream, cudaMemcpyHostToDevice, s[is]) );  
          CUDA_SAFE_CALL( cudaMemcpyAsync((void*) &d_offs[a1], (void*) &h_o[a2], memSizeStream, cudaMemcpyHostToDevice, s[is]) );
        }
#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(mem_i_e, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(mem_i_e) );
#endif
        /*** ----  main compute kernel ----- ***/
        /*** this is where the magic happens ***/
#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(ker_b, 0) );
#endif
        for(int is=0; is<nStream; is++){
          a1 = is*Offset;
          my_voigt<<<dimGrid, dimBlock, 0, s[is]>>>(&d_damp[a1], &d_offs[a1], &d_vval[a1]);
        }
#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(ker_e, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(ker_e) );
#endif

        /* transfer data GPU -> CPU */
#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(mem_o_b, 0) );
#endif
        for(int is=0; is<nStream; ++is){
          a1 = is*Offset;
          a2 = startElemChunk + a1;
          CUDA_SAFE_CALL( cudaMemcpyAsync((void*) &h_v[a2], (void*) &d_vval[a1], memSizeStream, cudaMemcpyDeviceToHost, s[is]) );
        }
#ifdef _TIMING_PROF_
        CUDA_SAFE_CALL( cudaEventRecord(mem_o_e, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(mem_o_e) );

        CUDA_SAFE_CALL( cudaEventElapsedTime(&temp, ker_b, ker_e) );
        ker_elapsed += temp;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&temp, mem_i_b, mem_i_e) );
        mem_i_elapsed += temp;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&temp, mem_o_b, mem_o_e) );
        mem_o_elapsed += temp;
#endif
      }

      /* stop timing */
      CUDA_SAFE_CALL( cudaEventRecord(tot_e, 0) );
      CUDA_SAFE_CALL( cudaEventSynchronize(tot_e) );

      /* destroy local streams */
      for(int is=0; is<nStream; ++is)
        CUDA_SAFE_CALL( cudaStreamDestroy(s[is]) );

      free(s);

      CUDA_SAFE_CALL( cudaFreeHost(h_o) );
      CUDA_SAFE_CALL( cudaFreeHost(h_d) );

      /* allocate array to hold answers, and copy from local arrays */
#pragma omp single
      h_vval = (float*) malloc(memSize);
      {
        for(int i=0; i<nRowOmp*PROBLEM_SIZE; ++i){
           h_vval[i + startElemOmp] = h_v[i];
        }
      }

      /* free host output array */
      CUDA_SAFE_CALL( cudaFreeHost(h_v) );

      /* free memory on local device */
      CUDA_SAFE_CALL( cudaFree(d_damp) );
      CUDA_SAFE_CALL( cudaFree(d_offs) );
      CUDA_SAFE_CALL( cudaFree(d_vval) );

      CUDA_SAFE_CALL( cudaEventElapsedTime(&tot_elapsed, tot_b, tot_e) );
      CUDA_SAFE_CALL( cudaThreadSynchronize() );

      // print a few values for verification
#pragma omp single
      {
      cout << endl << "Verification values:"<<endl;
      cout         << "-------------------"<<endl;
      for(int i=PROBLEM_SIZE/2; i<PROBLEM_SIZE/2 + 5; i++){
        for(int j=0; j<2; j++){
          cout << h_vval[i + j*PROBLEM_SIZE] << " ";
        }
        cout << endl;
      }
      cout << "-------------------"<<endl;
      }


      /* print timing results */
#pragma omp single
      {
        cout << "-----------------------------------------" << endl;
        cout << "Elapsed times (msec): "                    << endl;
      }
#pragma omp barrier
#pragma omp critical
      {
        cout << "    Thread " << ompTid << ":" << endl;
        cout << "      - total:          " << tot_elapsed << endl;
#ifdef _TIMING_PROF_
        cout << "      - compute kernel: " << ker_elapsed << endl;
        cout << "      - mem xfer total: " << mem_i_elapsed + mem_o_elapsed << endl;
        cout << "                    in: " << mem_i_elapsed << endl;
        cout << "                   out: " << mem_o_elapsed << endl;
#endif        
      }
#pragma omp barrier
#pragma omp single
     {
       cout << "-----------------------------------------" << endl;
     }

#ifdef _OPENMP
    }
#endif


  /* release memory */
  free(h_vval);
    
}
