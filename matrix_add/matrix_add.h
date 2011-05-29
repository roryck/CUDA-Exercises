#ifndef _MATRIX_ADD_H_
#define _MATRIX_ADD_H_

/*
 * Kernel routine to add to matrices
 */
#include "dims.h"
__global__ void matrix_add(float *m1, float *m2, float *mout);
__global__ void matrix_add_2d(float *m1, float *m2, float *mout);
#endif
