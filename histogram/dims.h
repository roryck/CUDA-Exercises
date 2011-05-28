#ifndef _DIMS_H_
#define _DIMS_H_

#include "../common/macros.h"

/* number of bins */
const int nbins = 10;

/* length of the vectors */
//const int len             = 234177;
const int len = 123456;

/* number of CUDA threads in each block */
const int threadsPerBlock =  32;

/* number of blocks in the grid */
const int blocksPerGrid   = min(30, (len + threadsPerBlock-1)/threadsPerBlock);

#endif
