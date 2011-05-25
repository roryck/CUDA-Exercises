#ifndef _ERRCK_H_
#define _ERRCK_H_

inline void errorCheck(cudaError_t err)
{
        if(err != cudaSuccess){
                printf("Error: %s\n", cudaGetErrorString(err));
        }
        return;
}

inline void errorCheck(const char *file, const int line)
{
#ifdef DEBUG
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("CUDA Error in file %s, line %i: %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}
	

#endif
