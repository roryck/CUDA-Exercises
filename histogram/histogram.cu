/* histogram.cu                                
 * Kernel routine to create a histogram from an
 * array of input data.  Within a given block each
 * thread creates a private histogram.  The histograms
 * are then reduced across threads to give one histogram
 * per block.  The block-level histograms are packed 
 * into an array an returned to the CPU for the final 
 * reduction step.
 * Input:  data
 * Output: histo
 */
#include "dims.h"

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
			
		
