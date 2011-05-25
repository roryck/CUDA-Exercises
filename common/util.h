#ifndef _UTIL_H_
#define _UTIL_H_

#include<stdio.h>
#include "errck.h"

void print_chars(char *c, int N)
{
	int i;
	for(i=0; i<N; ++i)
		printf("%c",c[i]);
	printf("\n");
	return;
}
	

void printDevInfo()
{
	int ndev;           // number of devices
	int i;              // index
	cudaError_t cErr;   // for error checking
	cudaDeviceProp dp; // pointer to cudaDeviceProp struct
	
        /* see how many devices are available */
	cErr = cudaGetDeviceCount(&ndev);
        errorCheck(cErr);
	printf("-----------------------------------\n");
	if(ndev == 1)
		printf("Found %d CUDA capable device\n", ndev);
	else
		printf("Found %d CUDA capable devices\n", ndev);

        printf("-----------------------------------\n");

        /* print some info about the available devices */
	for(i=0; i<ndev; ++i){
		/* get some properties of device i */
		cudaGetDeviceProperties(&dp, i);
		printf("Device %i:\n", i);
		printf("  %s\n", dp.name);
                printf("  Compute Capability: %d.%d\n", dp.major, dp.minor);
		printf("  %i multiprocessors at %i MHz\n", dp.multiProcessorCount, dp.clockRate/1000);
		printf("  %6.2f MB Global Memory\n", float(dp.totalGlobalMem)/1048576.0);
	}
	errorCheck(__FILE__,__LINE__);
        printf("-----------------------------------\n");
	
	return;
}

#endif /* _UTIL_H_ */
