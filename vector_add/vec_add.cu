/* vec_add.cu                                
 * Kernel routine to perform vector addition on 2 vectors
 * of length len.
 * Input:  v1, v2
 * Output: v3
 */
#include "vec_add.h"
__global__ void vec_add(float *v1, float *v2, float *v3, size_t len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < len)
		v3[tid] = v1[tid] + v2[tid];
	return;
}
