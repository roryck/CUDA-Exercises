#ifndef _DOT_PROD_H_
#define _DOT_PROD_H_
/* dot_prod.h                                
 * Kernel routine to calculate dot product of
 * 2 vectors of length len.
 * Input:  v1, v2
 * Output: psum
 */
__global__ void dot_prod(float *v1, float *v2, float *psum);
#endif
