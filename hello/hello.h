#ifndef _HELLO_H_
#define _HELLO_H_
/* hello.h                                         *
 * Three simple CUDA kernels used to demonstrate   *
 * the basics of host/device data transfer kernel  *
 * invocation, and thread/block/grid layout in 1d  */
__global__ void hello(char *a, int N);
__global__ void hello_thread(char *a, int N);
__global__ void hello_both(char *a, int N);
#endif
