/* main.cu 
 * ----------------------------------------------------
 * Main routine for the dot-product example from the 
 * summer 2011 Intro to CUDA course taught at NCAR.
 * Author: Rory Kelly (rory@ucar.edu)
 * Created: 8 March 2011
 * ----------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "../common/util.h"
#include "dims.h"
#include "dot_prod.h"

/* local function declarations */

int main(int argc, char **argv)
{

	/* data on the CPU */
	float *h_vec1;      // input vector 
	float *h_vec2;      // input vector
        float *h_part;      // partial result
	float dp;           // dot product

	/* arrays to hold the vectors on GPU */
	float *d_vec1;      // input vector
	float *d_vec2;      // input vector
	float *d_part;      // partial result

	/* allocate CPU memory */
	h_vec1 = (float *) malloc(len*sizeof(float));
	h_vec2 = (float *) malloc(len*sizeof(float));
	h_part = (float *) malloc(blocksPerGrid*sizeof(float));

	/* allocate GPU memory */
	cudaMalloc((void **) &d_vec1, len*sizeof(float));
	cudaMalloc((void **) &d_vec2, len*sizeof(float));
	cudaMalloc((void **) &d_part, blocksPerGrid*sizeof(float));

	/* local vars */
	int idx;

        /* get some basic info about available devices */
        printDevInfo();
	
	/* initialize vectors on CPU */
        /* dot product should sum to */
        /* len / 2.0                 */
	for(idx=0; idx<len; idx++){
		h_vec1[idx] = 0.5f;
		h_vec2[idx] = 1.0f;
	}

        /* copy memory to device array */
	cudaMemcpy(d_vec1, h_vec1, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec2, h_vec2, len*sizeof(float), cudaMemcpyHostToDevice);

	/* call kernel */
	dot_prod<<<blocksPerGrid, threadsPerBlock>>>(d_vec1, d_vec2, d_part);

	/* copy data back to host */
	cudaMemcpy(h_part, d_part, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

	/* complete sum on CPU */
	dp=0.0;
	for(idx=0; idx<blocksPerGrid; idx++)
		dp+=h_part[idx];
		
	/* print results */
	printf("----------------------------------------------\n");
	printf("Vector length: %d\n", len);
	printf("Expected value:   %8.2f\n", ((float)len)/2.0);
	printf("Calcualted value: %8.2f\n", dp);
        printf("----------------------------------------------\n");

	
        /* clean up memory on host and device */
	cudaFree(d_vec1);
	cudaFree(d_vec2);
	cudaFree(d_part);
	free(h_vec1);
	free(h_vec2);
	free(h_part);

	return(0);
}
