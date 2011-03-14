/* hello CUDA kernel 
 * Arguments:
 *   char *a  - an array of characters
 * Purpose:
 *   Each CUDA thread calculates an index value and increments
 *   its portion of the array by the value of its index.
 */
__global__ void hello(char *a)
{
	int i = threadIdx.x;
	a[i] = a[i] + i;
}
