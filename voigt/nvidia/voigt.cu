#include <stdio.h>
#include "voigt.h"

__global__ void my_voigt(float *damp_arr, float *offs_arr, float *voigt_value){

  /* coefficients of the rational approximation formula */
  float A0 = 122.607931777104326f;
  float A1 = 214.382388694706425f;
  float A2 = 181.928533092181549f;
  float A3 = 93.155580458138441f;
  float A4 = 30.180142196210589f;
  float A5 = 5.912626209773153f;
  float A6 = 0.564189583562615f;
  float B0 = 122.60793177387535f;
  float B1 = 352.730625110963558f;
  float B2 = 457.334478783897737f;
  float B3 = 348.703917719495792f;
  float B4 = 170.354001821091472f;
  float B5 = 53.992906912940207f;
  float B6 = 10.479857114260399f;

  int ivsigno;
  float V;
  float Z1_real ;
  float Z1_imag;
  float Z2_real;
  float Z2_imag;
  float Z3_real;
  float Z3_imag;
  float Z4_real;
  float Z4_imag;
  float Z5_real;
  float Z5_imag;
  float Z6_real;
  float Z6_imag;
  float ZZ1_real;
  float ZZ1_imag;
  float ZZ2_real;
  float ZZ2_imag;
  float ZZ3_real;
  float ZZ3_imag;
  float ZZ4_real;
  float ZZ4_imag;
  float ZZ5_real;
  float ZZ5_imag;
  float ZZ6_real;
  float ZZ6_imag;
  float ZZ7_real;
  float ZZ7_imag;
  float division_factor;
  float ZZZ_real;

  /* calculate thread index */
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  /* get local values of damping and offset */
  float damping = damp_arr[idx];
  float offset = offs_arr[idx];

  if (offset < 0)                                       
    ivsigno=-1;                                    
  else                                                            
    ivsigno=1;                                     
  V = ivsigno * offset;       
  Z1_real = A6 * damping + A5; 
  Z1_imag = A6 * -V;   
  Z2_real = Z1_real * damping - Z1_imag * -V + A4; 
  Z2_imag = Z1_real * -V + Z1_imag * damping; 
  Z3_real = Z2_real * damping - Z2_imag * -V + A3; 
  Z3_imag = Z2_real * -V + Z2_imag * damping; 
  Z4_real = Z3_real * damping - Z3_imag * -V + A2; 
  Z4_imag = Z3_real * -V + Z3_imag * damping; 
  Z5_real = Z4_real * damping - Z4_imag * -V + A1; 
  Z5_imag = Z4_real * -V + Z4_imag * damping; 
  Z6_real = Z5_real * damping - Z5_imag * -V + A0; 
  Z6_imag = Z5_real * -V + Z5_imag * damping; 
  ZZ1_real = damping + B6;          
  ZZ1_imag = -V;                    
  ZZ2_real = ZZ1_real * damping - ZZ1_imag * -V  + B5; 
  ZZ2_imag = ZZ1_real * -V + ZZ1_imag * damping; 
  ZZ3_real = ZZ2_real * damping - ZZ2_imag * -V  + B4; 
  ZZ3_imag = ZZ2_real * -V + ZZ2_imag * damping; 
  ZZ4_real = ZZ3_real * damping - ZZ3_imag * -V  + B3; 
  ZZ4_imag = ZZ3_real * -V + ZZ3_imag * damping; 
  ZZ5_real = ZZ4_real * damping - ZZ4_imag * -V  + B2; 
  ZZ5_imag = ZZ4_real * -V + ZZ4_imag * damping; 
  ZZ6_real = ZZ5_real * damping  - ZZ5_imag * -V + B1; 
  ZZ6_imag = ZZ5_real * -V  + ZZ5_imag * damping; 
  ZZ7_real = ZZ6_real * damping  - ZZ6_imag * -V + B0; 
  ZZ7_imag = ZZ6_real * -V  + ZZ6_imag * damping; 
  division_factor = 1.0f / (ZZ7_real * ZZ7_real + ZZ7_imag * ZZ7_imag); 
  ZZZ_real = (Z6_real * ZZ7_real  + Z6_imag * ZZ7_imag) * division_factor; 
  // not returning faraday value so ZZZ_imag does not need to be calculated
  // ZZZ_imag = (Z6_real * -ZZ7_imag + Z6_imag * ZZ7_real) * division_factor; 
  voigt_value[idx] = ZZZ_real;;                       
}
