#ifndef _DIMS_H_
#define _DIMS_H_

#include "../common/macros.h"

/* length of the vectors */
const int len             = 234177;

/* number of CUDA threads in each block */
const int threadsPerBlock =  32;

/* number of blocks in the grid */
const int blocksPerGrid   = min(64, (len + threadsPerBlock-1)/threadsPerBlock);

#endif
