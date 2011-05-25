#ifndef _VEC_ADD_H_
#define _VEC_ADD_H_
/* vec_add.h                                
 * Kernel routine to perform vector addition on 2 vectors
 * of length len.
 * Input:  v1, v2
 * Output: v3
 */
__global__ void vec_add(float *v1, float *v2, float *v3, int len);
#endif
