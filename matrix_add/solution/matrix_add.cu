/*
 * Kernel routine to add to matrices
 */
#include "dims.h"
__global__ void matrix_add(float *m1, float *m2, float *mout)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < size*size){
		mout[tid] = m1[tid]+m2[tid];
		tid += blockDim.x;
	}
}

__global__ void matrix_add_2d(float *m1, float *m2, float *mout)
{	
	int tid = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.x;
	while(tid < size*size){
		mout[tid] = m1[tid]+m2[tid];
		tid += blockDim.x * blockDim.y;
	}
}
