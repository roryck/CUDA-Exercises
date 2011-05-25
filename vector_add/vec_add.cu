/* vec_add.cu                                
 * Kernel routine to perform vector addition on 2 vectors
 * of length len.
 * Input:  v1, v2
 * Output: v3
 */
#include "vec_add.h"
__global__ void vec_add(float *v1, float *v2, float *v3, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i;

	if(tid == 0){
		for(i=0; i<len; i++)
			v3[i] = v1[i] + v2[i];
	}
}
