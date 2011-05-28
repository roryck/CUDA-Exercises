/* dot_prod.cu                                
 * Kernel routine to calculate dot product of
 * 2 vectors of length len.
 * Input:  v1, v2
 * Output: psum
 */
#include "dims.h"

__global__ void dot_prod(float *v1, float *v2, float *psum)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r_idx;
	float threadSum = 0.0f;
	__shared__ float loc_sum[threadsPerBlock];

	while(tid < len){
		threadSum += v1[tid] * v2[tid];
		tid += blockDim.x * gridDim.x;
	}
	loc_sum[threadIdx.x] = threadSum;

	__syncthreads();

	r_idx = blockDim.x / 2;
	while(r_idx != 0) {
		if(threadIdx.x < r_idx)
			loc_sum[threadIdx.x] += loc_sum[threadIdx.x + r_idx];
		__syncthreads();
		r_idx /= 2;
	}
	if(threadIdx.x == 0) psum[blockIdx.x] = loc_sum[0];
}

__global__ void histogram(float *data, int *histo)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int r_idx;
	int t_hist[nbins];                              // thread local histogram
	__shared__ int b_hist[threadsPerBlock][nbins];  // block local histogram

	/* initialize histograms to zero */
	for(int i=0; i<nbins; i++){
		t_hist[i] = 0;
		b_hist[tid][i] = 0;
	}
	while(tid < len){
		t_hist[(int)(data[tid])]++;
		tid += blockDim.x * gridDim.x;
	}
	for(int i=0; i<nbins; i++) b_hist[threadIdx.x][i] = t_hist[i];

	__syncthreads();

	r_idx = blockDim.x / 2;
	while(r_idx != 0){
		if(threadIdx.x < r_idx){
			for(int i=0; i<nbins; i++)
				b_hist[threadIdx.x][i] += b_hist[threadIdx.x + r_idx][i];
		}
		__syncthreads();
		r_idx /= 2;
	}
	if(threadIdx.x == 0){
		for(int i=0; i<nbins; i++)
			histo[i + nbins*blockIdx.x] = b_hist[threadIdx.x][i];
	}
}
			
		
