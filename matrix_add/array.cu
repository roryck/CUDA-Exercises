/* array.c
 * Utilities used to work with basic arrays in the
 * CUDA Example course.  Array data is assumed to 
 * be type float unless otherwise noted.
 */

#include<stdlib.h>
#include<stdio.h>
#include "array.h"
#include "dims.h"

/*
void alloc_2d(float ***array, int nr, int nc)
{
        float *tmp;
        int i;
        tmp = (float *) malloc(nc*nr*sizeof(float));
        *array = (float **) malloc(nr*sizeof(float *));
        for(i=0;i<nr;i++)
                *(*array+i) = tmp+i*nc;
}
*/

void alloc_2d(float ***array, int nr, int nc)
{
	int i;
	float *tmp = (float *) malloc(nr*nc*sizeof(float));
	*array = (float **) malloc(nr*sizeof(float *));
	for(i=0;i<nr;i++){
		//printf("%p : %p \n", (*array)[i], &(tmp[i*nc]));
		(*array)[i] = &tmp[i*nc];
	}
		
}

void dealloc_2d(float ***array)
{
	free( (*array)[0] );
	free( *array );
}


/* print_2d
 * routine to print the contents of a 2d array 
 */
void print_2d(float **a, int nr, int nc)
{
	int i,j;
	for(i=0; i<nr; i++){
		for(j=0; j<nc; j++){
			printf("%6.2f ", a[i][j]);
		}
		printf("\n");
	}
}
