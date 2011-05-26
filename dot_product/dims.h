#ifndef _DIMS_H_
#define _DIMS_H_

#include "../common/macros.h"

const int len             = 234177;
const int threadsPerBlock =  32;
const int blocksPerGrid   = min(64, (len + threadsPerBlock-1)/threadsPerBlock);

#endif
