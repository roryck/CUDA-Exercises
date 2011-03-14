#ifndef _HELLO_H_
#define _HELLO_H_
/* hello.h                                     *
 * A simple CUDA kernel used to demonstrate    *
 * the basics of host/device data transfer and *
 * kernel invocation                           */
__global__ void hello(char *a);

#endif
