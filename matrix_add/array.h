#ifndef _ARRAY_H_
#define _ARRAY_H_
#include "dims.h"
/* array.h
 * Utilities used to work with basic arrays in the
 * CUDA Example course.  Array data is assumed to 
 * be type float unless otherwise noted.
 */

/* alloc_2d
 * routine to allocate a contiguous
 * 2d array of floats
 */ 
void alloc_2d(float ***array, int nr, int nc);

/* dealloc_2d
 * routine to free the memory associted with
 * and array created by alloc_2d
 */
void dealloc_2d(float **array);

/* print_2d
 * routine to print the contents of a 2d array 
 */
void print_2d(float a[size][size], int nr, int nc);

void testfun(unsigned int n);
#endif
