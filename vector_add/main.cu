/* hello.cu
 * A Hello World example in CUDA
 * ----------------------------------------------------
 * Example 1 from the summer 2011 Intro to CUDA course 
 * taught at NCAR.  
 * Author: Rory Kelly (rory@ucar.edu)
 * Created: 8 March 2011
 * ----------------------------------------------------
 * Example program showing the summation of two vectors
 * on the GPU.
 * ----------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/util.h"
#include "vec_add.h"

/* function declarations */

void vec_print(float *v, int len);
void vector_add(float *v1, float *v2, float *v3, int len);

int main(int argc, char **argv)
{

	/* vector length */
	int len=12;

	/* data on the CPU to be added */
	float *h_vec1;
	float *h_vec2;
	float *h_vec3;
	float *result;

	/* arrays to hold the vectors on GPU */
	float *d_vec1;
	float *d_vec2;
	float *d_vec3;

	/* allocate CPU memory */
	h_vec1 = (float *) malloc(len*sizeof(float));
	h_vec2 = (float *) malloc(len*sizeof(float));
	h_vec3 = (float *) malloc(len*sizeof(float));
	result = (float *) malloc(len*sizeof(float));

	/* allocate GPU memory */
	cudaMalloc((void **) &d_vec1, len*sizeof(float));
	cudaMalloc((void **) &d_vec2, len*sizeof(float));
	cudaMalloc((void **) &d_vec3, len*sizeof(float));

	/* local vars */
	int idx;

        /* get some basic info about available devices */
        printDevInfo();
	
	/* initialize vectors on CPU */
	for(idx=0; idx<len; idx++){
		h_vec1[idx] = (float)idx;
		h_vec2[idx] = (float)(idx*idx);
	}

	/* perform sum on CPU for validation */
	vector_add(h_vec1, h_vec2, result, len);

        /* copy memory to device array */
	cudaMemcpy(d_vec1, h_vec1, len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec2, h_vec2, len, cudaMemcpyHostToDevice);

	/* call kernel */
	vec_add<<<20,20>>>(d_vec1, d_vec2, d_vec3, len);

	/* copy data back to host */
	cudaMemcpy(h_vec3, d_vec3, len, cudaMemcpyDeviceToHost);

	/* print contents of arrays */
	vec_print(result, len);
	vec_print(h_vec3, len);
	
        /* clean up memory on host and device */
	//cudaFree(d_data);
	free(h_vec1);
	free(h_vec2);
	free(h_vec3);

	return(0);
}

/* Routine to add 2 vectors of length  len and return the 
 * result in a third vector.
 * Input vectors: v1, v2
 * Output vector: v3
 */
void vector_add(float *v1, float *v2, float *v3, int len)
{
	int i;
	for (i=0; i<len; i++)
		v3[i] = v1[i] + v2[i];
	return;
}

/* routine to print contents of vector */
void vec_print(float *v, int len)
{
	int i;
	for(i=0; i<len; i++)
		printf("%6.2f ", v[i]);
	printf("\n");
}
