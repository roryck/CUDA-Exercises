__global__ void hello(char *a)
{
	int i = threadIdx.x;
	a[i] = a[i] + i;
}
