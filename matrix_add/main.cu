/* main.cu
 * An array copy demo in CUDA
 * ---------------------------------------------------
 * Example 2 from the summer 2011 Intro to CUDA course 
 * taught at NCAR.  
 * Author: Rory Kelly (rory@ucar.edu)
 * Created: 8 March 2011
 * ---------------------------------------------------
 * This is a short program demonstrating the copying
 * of a 2D array from Host to Device memory.
 * ---------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/util.h"
#include "array.h"
#include "dims.h"
#include "matrix_add.h"

int main(int argc, char **argv)
{

	/* 2D arrays that will be allocated on host */
	float **h_data_in1;
	float **h_data_in2;
	float **h_data_out;

	/* 1D array on GPU that will hold CPU data */
	float *d_data1;
	float *d_data2;
	float *d_out;

	/* local vars */
	dim3 blockDims;     // used for 2d-indexed kernel
	float resultTrace;  // used for result validation

        /* get some basic info about available devices */
        printDevInfo();

	/* print info about the problem layout */
	printf("Adding two matrices of size: %d x %d\n", size, size);
#ifdef ONE_D_INDEX
	printf("Threads per block: %d\n", threadsPerBlock);
#else
	printf("Threads per block: %d (%d x %d)\n", threadsPerBlock, thdsX, thdsY);
#endif
	printf("Blocks per grid:   %d\n", blocksPerGrid); 
	
        printf("--------------------------------------\n");
	
	/* allocate and initialize 2d host arrays */
	alloc_2d(&h_data_in1, size, size);
	alloc_2d(&h_data_in2, size, size);
	alloc_2d(&h_data_out, size, size);

	/* initialize arrays so they sum to the identity matrix */
	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			h_data_in1[i][j] = 1.0f + j+i*size;
			h_data_in2[i][j] = -1.0f - (j+i*size);
			h_data_out[i][j] = 0.0f;
			if(i == j) h_data_in2[i][j] += 1.0f; 
		}
	}

	/* allocate the GPU arrays */
        cudaMalloc(&d_data1, size*size*sizeof(float));
	cudaMalloc(&d_data2, size*size*sizeof(float));
	cudaMalloc(&d_out, size*size*sizeof(float));

	/* transfer data to GPU */
	/* --- transfer whole matrices --- */
	cudaMemcpy(d_data1, h_data_in1[0], size*size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data2, h_data_in2[0], size*size*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_out, h_data_out[0], size*size*sizeof(float), cudaMemcpyHostToDevice);   // transfer 0 matrix to output array to clear results
	/* --- or transfer by rows --- */
	//for(int i=0; i<size; i++){
	//	cudaMemcpy(&d_data1[i*size], &h_data_in1[0][i*size], size*sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(&d_data2[i*size], &h_data_in2[0][i*size], size*sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(&d_out[i*size], &h_data_out[0][i*size], size*sizeof(float), cudaMemcpyHostToDevice);
	//}

	/* launch kernel */
#ifdef ONE_D_INDEX
	matrix_add<<<threadsPerBlock,blocksPerGrid>>>(d_data1, d_data2, d_out); // 1d-indexed kernel
#else
	blockDims.x = thdsX;
	blockDims.y = thdsY;
	matrix_add_2d<<<blockDims,blocksPerGrid>>>(d_data1, d_data2, d_out);       // 2d-indexed kernel
#endif

	/* transfer data back to CPU */
        /* --- transfer whole matrices --- */
	cudaMemcpy(h_data_out[0], d_out, size*size*sizeof(float), cudaMemcpyDeviceToHost);
	/* --- or transfer by rows --- */
	//for(int i=0; i<size; i++){
        //        cudaMemcpy(&h_data_out[i][0], &d_out[size*i], size*sizeof(float), cudaMemcpyDeviceToHost);
        //}

	/* print contents of arrays if they are small enough */
	if(size <= 20){
		print_2d(h_data_in1, size, size);
        	printf("\n");
		print_2d(h_data_out, size, size);
		printf("\n");
	}

	/* calculate the trace of the result matrix (expected answer = size) */
	resultTrace = 0.0f;
	for(int i=0; i<size; i++)
		resultTrace += h_data_out[i][i];

	/* compare result with expected answer */
	printf("Expected Trace:   %6.2f\n", (float)size);
	printf("Calculated Trace: %6.2f\n", resultTrace);
	
        /* clean up memory on host and device */
	dealloc_2d(&h_data_in1);
	dealloc_2d(&h_data_in2);
	dealloc_2d(&h_data_out);
	cudaFree(d_data1);
	cudaFree(d_data2);
	cudaFree(d_out);

	/* return and exit */
	return(0);
}

