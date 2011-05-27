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
//#include "../common/util.h"
#include "array.h"
#include "dims.h"

int main(int argc, char **argv)
{

	/* 2D arrays that will be allocated on host */
	//float **h_data_in  = NULL;
	//float **h_data_out = NULL;
	float h_data_in[size][size];
	float h_data_out[size][size];

	/* 1D array on GPU that will hold CPU data */
	float *d_data;

        /* get some basic info about available devices */
        //printDevInfo();
	
	/* allocate and initialize array */
	//alloc_2d(&h_data_in, size, size);
	//alloc_2d(&h_data_out, size, size);

	/* init array */
	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			h_data_in[i][j] = 1.0f+ j+i*size;
		}
	}

	/* allocate the GPU array */
        cudaMalloc(&d_data, size*size*sizeof(float));

	/* transfer data to GPU */
	cudaMemcpy(d_data, h_data_in, size*size*sizeof(float), cudaMemcpyHostToDevice);
	//for(int i=0; i<size; i++){
	//	cudaMemcpy(&d_data[i], &h_data_in[0][i], size*sizeof(float), cudaMemcpyHostToDevice);
	//}

	/* call kernel */
	cudaThreadSynchronize();

	/* transfer data back to CPU */
	cudaMemcpy(h_data_out, d_data, size*size*sizeof(float), cudaMemcpyDeviceToHost);
	//for(int i=0; i<size; i++){
        //        cudaMemcpy(&h_data_out[0][i], &d_data[i], size*sizeof(float), cudaMemcpyDeviceToHost);
        //}

	/* print contents of array */
	print_2d(h_data_in, size, size);
        printf("\n");

	print_2d(h_data_out, size, size);
	printf("\n");
	
        /* clean up memory on host and device */
	//dealloc_2d(h_data_in);
	//dealloc_2d(h_data_out);
	return(0);
}

