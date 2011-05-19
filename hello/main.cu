/* hello.cu
 * A Hello World example in CUDA
 * ---------------------------------------------------
 * Example 1 from the summer 2011 Intro to CUDA course 
 * taught at NCAR.  
 * Author: Rory Kelly (rory@ucar.edu)
 * Created: 8 March 2011
 * ---------------------------------------------------
 * This is a short program which uses multiple threads
 * CUDA threads to calculate a "Hello World" message
 * which is then printed to the screen.  It's intended
 * to demonstrate the execution of a CUDA kernel, and
 * show CUDA threads executing independently to 
 * calculate a result.
 * ---------------------------------------------------
 */
#define SIZE 12
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "util.h"
#include "hello.h"

int main(int argc, char **argv)
{
	/* data that will live on host */
	char *data; // data array

	/* data that will live in device memory */
	char *d_data;

        /* get some basic info about available devices */
        printDevInfo();
	
	/* allocate and initialize array */
	data = (char*) malloc(SIZE*sizeof(char));
	data[0]  =  72; data[1]  = 100;
	data[2]  = 106; data[3]  = 105;
	data[4]  = 107; data[5]  =  27;
	data[6]  =  81; data[7]  = 104;
	data[8]  = 106; data[9]  =  99;
	data[10] =  90; data[11] =  22; 

	/* print data before kernel call */
	printf("Contents of data before kernel call: ");
	print_chars(data, SIZE);

	/* allocate memory on device */
	cudaMalloc(&d_data, SIZE*sizeof(char));

        /* copy memory to device array */
	cudaMemcpy(d_data, data, SIZE, cudaMemcpyHostToDevice);

	/* call kernel with one block / SIZE threads */
        // hello_block<<<15, 1>>>(d_data, SIZE);
        //hello_thread<<<1,15>>>(d_data, SIZE);
	hello_both<<<4,3>>>(d_data, SIZE);

	/* copy data back to host */
	cudaMemcpy(data, d_data, SIZE, cudaMemcpyDeviceToHost);

	/* print contents of array */
        printf("Contents of data after kernel call:  ");
	print_chars(data, SIZE);
	printf("\n");
	
        /* clean up memory on host and device */
	cudaFree(d_data);
	free(data);
	return(0);
}

