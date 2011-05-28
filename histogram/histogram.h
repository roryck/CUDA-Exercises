#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

/* histogram.cu
 * Kernel routine to create a histogram from an
 * array of input data.  Within a given block each
 * thread creates a private histogram.  The histograms
 * are then reduced across threads to give one histogram
 * per block.  The block-level histograms are packed 
 * into an array an returned to the CPU for the final 
 * reduction step.
 * Input:  data
 * Output: histo
 */
__global__ void histogram(float *data, int *histo);

#endif
