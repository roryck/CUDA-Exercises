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
#include "histogram.h"

/* local function declarations */

int main(int argc, char **argv)
{

	/* array of floats x: 0.0 <= x < 10.0 */
	float *h_data;     // arrays with data to create histogram
	int *h_hist;       // partially reduced histogram of float data
	int *gpu_ans;      // final histogram from GPU
	int *cpu_ans;      // histogram on CPU for validation

	/* locals */
	int correct;

	/* arrays to hold the vectors on GPU */
	float *d_data;     // input vector
	int *d_hist;       // partial result

	/* get some basic info about available devices */
        printDevInfo();

	/* print info about the problem layout */
	printf("--------------------------------------\n");
        printf("Creating histogram from %d data points\n", len);
	printf("Computational Grid Layout:\n");
        printf("  Threads per block: %d\n", threadsPerBlock);
        printf("  Blocks per grid:   %d\n", blocksPerGrid);
        printf("--------------------------------------\n");

	/* allocate CPU memory */
	h_data = (float *) malloc(len*sizeof(float));
	/* Each block will return a histogram.  The CPU will */
	/* produce the final histogram.                      */
	h_hist = (int *) malloc(nbins*blocksPerGrid*sizeof(int));
	cpu_ans = (int *) malloc(nbins*sizeof(int));
	gpu_ans = (int *) malloc(nbins*sizeof(int));

	/* initialize histogram arrays */
	for(int i=0; i<nbins*blocksPerGrid; i++){
		if(i < nbins){
			cpu_ans[i]=0;
			gpu_ans[i]=0;
		}
		h_hist[i]=0;
	}

	/* allocate GPU memory */
	cudaMalloc((void **) &d_data, len*sizeof(float));
	cudaMalloc((void **) &d_hist, nbins*blocksPerGrid*sizeof(int));

	/* initialize data on CPU by filling array with */
	/* random floats between 0 and 10               */
	srand(555);
	for(int i=0; i<len; i++){
		h_data[i] = 10.0f * ((float)rand())/(1.0f + (float)RAND_MAX);
	}

	if(len < 12){
		printf("data:\n");	
		for(int i=0; i<len; i++)
			printf("%8.7f\n",h_data[i]);
	}

        /* copy memory to device array */
	cudaMemcpy(d_data, h_data, len*sizeof(float), cudaMemcpyHostToDevice);

	/* call kernel */
	//dot_prod<<<blocksPerGrid, threadsPerBlock>>>(d_vec1, d_vec2, d_part);
	histogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist);

	/* copy data back to host */
	cudaMemcpy(h_hist, d_hist, nbins*blocksPerGrid*sizeof(int), cudaMemcpyDeviceToHost);

	/* reduce final histogram on CPU */
	for(int i=0; i<nbins; i++){
		for(int j=0; j<blocksPerGrid; j++){
			gpu_ans[i] += h_hist[i + nbins*j];
		}
	}

	/* calculate histogram on CPU for validation */
	for(int i=0; i<len; i++)
		cpu_ans[(int)(h_data[i])]++;
		
	/* print results */
	printf("--------------------- Histogram Comparison -----------------------\n");
	printf("                  CPU:              GPU:       \n");
	printf("[0-1):         %7d           %7d        \n", cpu_ans[0],gpu_ans[0]);
        printf("[1-2):         %7d           %7d        \n", cpu_ans[1],gpu_ans[1]);
        printf("[2-3):         %7d           %7d        \n", cpu_ans[2],gpu_ans[2]);
        printf("[3-4):         %7d           %7d        \n", cpu_ans[3],gpu_ans[3]);
        printf("[4-5):         %7d           %7d        \n", cpu_ans[4],gpu_ans[4]);
        printf("[5-6):         %7d           %7d        \n", cpu_ans[5],gpu_ans[5]);
        printf("[6-7):         %7d           %7d        \n", cpu_ans[6],gpu_ans[6]);
        printf("[7-8):         %7d           %7d        \n", cpu_ans[7],gpu_ans[7]);
        printf("[8-9):         %7d           %7d        \n", cpu_ans[8],gpu_ans[8]);
        printf("[9-10):        %7d           %7d        \n", cpu_ans[9],gpu_ans[9]);
        printf("------------------------------------------------------------------\n");
	correct=1;
	for(int i=0; i<nbins; i++)
		correct *= (cpu_ans[i] == gpu_ans[i]);
	if(correct)
		printf("CPU and GPU histograms are the same\n");
	else
		printf("Error: CPU and GPU histograms are not the same\n");
	printf("------------------------------------------------------------------\n");

        /* clean up memory on host and device */
	cudaFree(d_data);
	cudaFree(d_hist);
	free(h_data);
	free(h_hist);
	free(cpu_ans);
	free(gpu_ans);

	return(0);
}
