#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include <float.h>

#include "voigt.h"

// The standard test case sizes
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 4096
#endif

using namespace std;

int main(int argc, char* argv[])
{
    /* events and timing variables */
    cudaEvent_t mem_i_b, mem_i_e;
    cudaEvent_t mem_o_b, mem_o_e;
    cudaEvent_t kernel_b, kernel_e;
    float mem_i_elapsed, mem_o_elapsed, kernel_elapsed;
    float mem_total, total;
    cudaEventCreate(&mem_i_b);
    cudaEventCreate(&mem_i_e);
    cudaEventCreate(&mem_o_b);
    cudaEventCreate(&mem_o_e);
    cudaEventCreate(&kernel_b);
    cudaEventCreate(&kernel_e);

    int i=0;
    int j=0;

    float vals;
    float step = 32.0/PROBLEM_SIZE;

    /* check for CUDA devices on host system *
     * and print device info if found        */
    int ndev;
    int dev;
    cudaGetDeviceCount(&ndev);
    struct cudaDeviceProp deviceProp;
    if (ndev == 0)
      printf("No CUDA devices found.\n");
    else if(ndev == 1)
      printf("\nFound %d CUDA device:\n", ndev);
    else
      printf("\nFound %d CUDA devices:\n", ndev);
    for(dev=0; dev<ndev; ++dev){
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
      printf("    Max Grid Dims (X  Y  Z):  %10d  %5d  %5d\n", deviceProp.maxGridSize[0],
                                                              deviceProp.maxGridSize[1],
                                                              deviceProp.maxGridSize[2]);
      printf("\n");
    }

    /* configure the problem decomposition */
    int nthds; // threads per block
    int ntblks_x; // blocks in x
    int ntblks_y; // blocks in y

    nthds = 32 ;   // # of threads in a block - won't compile until value is set
    ntblks_x = PROBLEM_SIZE;  // # of blocks in the grid - won't compile until value is set
    ntblks_y = PROBLEM_SIZE/nthds;

    dim3 dimGrid(ntblks_x,ntblks_y);
    dim3 dimBlock(nthds);

    cout << "GPU Grid Decomposition:" << endl;
    cout << "  " << PROBLEM_SIZE*PROBLEM_SIZE << " total points " << endl;
    cout << "  " << ntblks_x << " thread blocks in X" << endl;
    cout << "  " << ntblks_y << " thread blocks in Y" << endl;
    cout << "  " << nthds << " threads per block" << endl;
 
    /* allocate space on host and device for input and output data */
    float *h_damp, *h_offs, *h_vval;
    float *d_damp, *d_offs, *d_vval;
    size_t memSize = PROBLEM_SIZE*PROBLEM_SIZE*sizeof(float);

    /* allocate host memory */
    h_damp = (float *) malloc(memSize);
    h_offs = (float *) malloc(memSize);
    h_vval = (float *) malloc(memSize);

    /* allocate device memory */
    cudaMalloc((void**)&d_damp, memSize);
    cudaMalloc((void**)&d_offs, memSize);
    cudaMalloc((void**)&d_vval, memSize);

    assert(FLT_EPSILON < step);

    vector <float> row;
    row.resize(PROBLEM_SIZE);

    /* assign initial data values */
    vals = 10.0;
    for(i=0; i<PROBLEM_SIZE; ++i){
        vals += step;
        for(j=0; j<PROBLEM_SIZE; ++j){
            h_damp[i*PROBLEM_SIZE + j] = vals;
            h_offs[j*PROBLEM_SIZE + i] = vals;
        }
    }
    
    /* transfer data CPU -> GPU */
    cudaEventRecord(mem_i_b, 0);
    cudaMemcpy((void*) d_damp, (void*) h_damp, memSize, cudaMemcpyHostToDevice);  
    cudaMemcpy((void*) d_offs, (void*) h_offs, memSize, cudaMemcpyHostToDevice);
    cudaEventRecord(mem_i_e, 0);
    cudaEventSynchronize(mem_i_e);

    /*** ----  main compute kernel ----- ***/
    /*** this is where the magic happens ***/
    cudaEventRecord(kernel_b, 0);

    my_voigt<<<dimGrid, dimBlock>>>(d_damp, d_offs, d_vval);

    cudaEventRecord(kernel_e, 0);
    cudaEventSynchronize(kernel_e);

    /* transfer data GPU -> CPU */
    cudaEventRecord(mem_o_b, 0);
    cudaMemcpy((void*) h_vval, (void*) d_vval, memSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(mem_o_e, 0);
    cudaEventSynchronize(mem_o_e);

    /* print verification values */
    cout << endl << "Verification values:"<<endl;
    cout         << "-------------------"<<endl;
    for(i=PROBLEM_SIZE/2; i<PROBLEM_SIZE/2 + 5; i++){
       for(j=0; j<2; j++){
          cout << h_vval[i*PROBLEM_SIZE + j] << " ";
       }
       cout << endl;
    }
    cout << "-------------------"<<endl;


    /* print information about elapsed time */
    cudaEventElapsedTime(&mem_i_elapsed, mem_i_b, mem_i_e);
    cudaEventElapsedTime(&kernel_elapsed, kernel_b, kernel_e);
    cudaEventElapsedTime(&mem_o_elapsed, mem_o_b, mem_o_e);
    mem_total = mem_i_elapsed + mem_o_elapsed;
    total = mem_total + kernel_elapsed;
    cout << "-----------------------------------------" << endl;
    cout << "Elapsed times (msec): "<< endl;
    cout << "    - memory xfer, cpu -> gpu: " << mem_i_elapsed << endl;
    cout << "    - memory xfer, gpu -> cpu: " << mem_o_elapsed << endl;
    cout << "    - memory total:            " << mem_total << endl;
    cout << "    - voigt kernel:            " << kernel_elapsed << endl;
    cout << "    - total:                   " << total << endl;
    cout << "-----------------------------------------" << endl;


}
