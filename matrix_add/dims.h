#ifndef _DIMS_H_
#define _DIMS_H_

#include "../common/macros.h"

/* for 1d indexed kernel define ONE_D_INDEX macro */
/* comment out for 2d indexing                    */
//#define ONE_D_INDEX

/* use arrays of dimension size x size */
const int size = 137;

/* number of cuda threads in a block */
#ifdef ONE_D_INDEX
const int threadsPerBlock = 32;
#else
const int thdsX = 16;                    // threads in X for 2d indexed kernel
const int thdsY =  4;                    // threads in Y for 2d indexed kernel
const int threadsPerBlock = thdsX*thdsY; // total threads per block;
#endif

/* number of blocks in the grid */
const int blocksPerGrid   = min(32, (size*size + threadsPerBlock-1)/threadsPerBlock);

#endif
