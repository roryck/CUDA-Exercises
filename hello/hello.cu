/* hello CUDA kernels
 * Arguments:
 *   char *a  - an array of characters
 * Purpose:
 *   Each CUDA thread calculates an index value and increments
 *   its portion of the array by the value of its index.
 */

/* hello_block
 * This kernel works when called with
 * multiple thread blocks, each using
 * a single thread
 */
__global__ void hello_block(char *a, int N)
{
	int i = blockIdx.x;
	if(i < N)
		a[i] = a[i] + i;
}

/* hello_thread
 * This kernel works when called with
 * a single thread block using
 * multiple threads
 */
__global__ void hello_thread(char *a, int N)
{
        int i = threadIdx.x;
	if(i < N)
		a[i] = a[i] + i;
}

/* hello_both
 * This kernel works when called with
 * multiple thread blocks each with
 * multiple threads
 */
__global__ void hello_both(char *a, int N)
{
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < N)
		a[i] = a[i] + i;
}
