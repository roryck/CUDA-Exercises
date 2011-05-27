#ifndef _DIMS_H_
#define _DIMS_H_

#include "../common/macros.h"

/* arrays of dimension size x size */
const int size = 825;

/* number of cuda threads in a block */
const int threadsPerBlock = 32;

/* number of blocks in the grid */
const int blocksPerGrid   = min(32, (size*size + threadsPerBlock-1)/threadsPerBlock);

#endif
