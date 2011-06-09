//#include <stdio.h>
#ifdef _SSE
#include <xmmintrin.h>
#endif
#ifdef __ALTIVEC__
// forward declaration of vector division
static inline vector float vec_div(vector float A, vector float B);
#endif
#include "Voigt.h"
#include <stdio.h>

#ifdef _SSE
#define VECLEN 4
/* A vectorized version of the Voigt function using X86 SSE instructions */
void my_voigt(const float *damping, const float *frequency_offset, float *voigt_value, int N)   
{                                                                   
    // coefficients of the rational approximation formula
    // to the complementary error function
    const __m128 A0 = _mm_set1_ps(122.607931777104326f);
    const __m128 A1 = _mm_set1_ps(214.382388694706425f);
    const __m128 A2 = _mm_set1_ps(181.928533092181549f);
    const __m128 A3 = _mm_set1_ps(93.155580458138441f);
    const __m128 A4 = _mm_set1_ps(30.180142196210589f);
    const __m128 A5 = _mm_set1_ps(5.912626209773153f);
    const __m128 A6 = _mm_set1_ps(0.564189583562615f);
    const __m128 B0 = _mm_set1_ps(122.60793177387535f);
    const __m128 B1 = _mm_set1_ps(352.730625110963558f);
    const __m128 B2 = _mm_set1_ps(457.334478783897737f);
    const __m128 B3 = _mm_set1_ps(348.703917719495792f);
    const __m128 B4 = _mm_set1_ps(170.354001821091472f);
    const __m128 B5 = _mm_set1_ps(53.992906912940207f);
    const __m128 B6 = _mm_set1_ps(10.479857114260399f);

    __m128 ivsigno;
    __m128 V;
    __m128 Z1_real;
    __m128 Z1_imag;
    __m128 Z2_real;
    __m128 Z2_imag;
    __m128 Z3_real;
    __m128 Z3_imag;
    __m128 Z4_real;
    __m128 Z4_imag;
    __m128 Z5_real;
    __m128 Z5_imag;
    __m128 Z6_real;
    __m128 Z6_imag;
    __m128 ZZ1_real;
    __m128 ZZ1_imag;
    __m128 ZZ2_real;
    __m128 ZZ2_imag;
    __m128 ZZ3_real;
    __m128 ZZ3_imag;
    __m128 ZZ4_real;
    __m128 ZZ4_imag;
    __m128 ZZ5_real;
    __m128 ZZ5_imag;
    __m128 ZZ6_real;
    __m128 ZZ6_imag;
    __m128 ZZ7_real;
    __m128 ZZ7_imag;
    __m128 division_factor;
    __m128 ZZZ_real;
    __m128 damp;
    __m128 offs;
    __m128 vval;
    __m128 one = _mm_set1_ps(1.0f); 
    __m128 zero = _mm_set1_ps(0.0f);
    __m128 mone = _mm_set1_ps(-1.0f);
    __m128 half = _mm_set1_ps(-0.5f);
    __m128 mask;

    float *stmp = (float *) _mm_malloc(4*sizeof(float), 16);

    int i;
    for(i=0; i<N; i+=VECLEN){
        _mm_prefetch((const char *)&damping[i+64], _MM_HINT_T0);
        _mm_prefetch((const char *)&frequency_offset[i+64], _MM_HINT_T0);
        damp = _mm_load_ps(&damping[i]);
        offs = _mm_load_ps(&frequency_offset[i]);
        mask = _mm_cmplt_ps(offs, zero);
        ivsigno = _mm_add_ps(_mm_and_ps(mask,mone),_mm_andnot_ps(mask,one));
        V = _mm_mul_ps(ivsigno, offs);       

        Z1_real = _mm_add_ps(_mm_mul_ps(A6, damp), A5);
        Z1_imag = _mm_mul_ps(A6, V);
        Z2_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(Z1_real, damp), _mm_mul_ps(Z1_imag, V)), A4);
        Z2_imag = _mm_add_ps(_mm_mul_ps(Z1_real, V), _mm_mul_ps(Z1_imag, damp));
        Z3_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(Z2_real, damp), _mm_mul_ps(Z2_imag, V)), A3);
        Z3_imag = _mm_add_ps(_mm_mul_ps(Z2_real, V), _mm_mul_ps(Z2_imag, damp));
        Z4_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(Z3_real, damp), _mm_mul_ps(Z3_imag, V)), A2);
        Z4_imag = _mm_add_ps(_mm_mul_ps(Z3_real, V), _mm_mul_ps(Z3_imag, damp));
        Z5_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(Z4_real, damp), _mm_mul_ps(Z4_imag, V)), A1);
        Z5_imag = _mm_add_ps(_mm_mul_ps(Z4_real, V), _mm_mul_ps(Z4_imag, damp));
        Z6_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(Z5_real, damp), _mm_mul_ps(Z5_imag, V)), A0);
        Z6_imag = _mm_add_ps(_mm_mul_ps(Z5_real, V), _mm_mul_ps(Z5_imag, damp));
        ZZ1_real = _mm_add_ps(damp, B6);          
        ZZ1_imag = V;                    
        ZZ2_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ1_real, damp), _mm_mul_ps(ZZ1_imag, V)), B5); 
        ZZ2_imag = _mm_add_ps(_mm_mul_ps(ZZ1_real, V), _mm_mul_ps(ZZ1_imag, damp)); 
        ZZ3_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ2_real, damp), _mm_mul_ps(ZZ2_imag, V)), B4); 
        ZZ3_imag = _mm_add_ps(_mm_mul_ps(ZZ2_real, V), _mm_mul_ps(ZZ2_imag, damp)); 
        ZZ4_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ3_real, damp), _mm_mul_ps(ZZ3_imag, V)), B3); 
        ZZ4_imag = _mm_add_ps(_mm_mul_ps(ZZ3_real, V), _mm_mul_ps(ZZ3_imag, damp)); 
        ZZ5_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ4_real, damp), _mm_mul_ps(ZZ4_imag, V)), B2); 
        ZZ5_imag = _mm_add_ps(_mm_mul_ps(ZZ4_real, V), _mm_mul_ps(ZZ4_imag, damp)); 
        ZZ6_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ5_real, damp), _mm_mul_ps(ZZ5_imag, V)), B1); 
        ZZ6_imag = _mm_add_ps(_mm_mul_ps(ZZ5_real, V), _mm_mul_ps(ZZ5_imag, damp)); 
        ZZ7_real = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ZZ6_real, damp), _mm_mul_ps(ZZ6_imag, V)), B0); 
        ZZ7_imag = _mm_add_ps(_mm_mul_ps(ZZ6_real, V), _mm_mul_ps(ZZ6_imag, damp)); 
        division_factor = _mm_div_ps(one, _mm_add_ps(_mm_mul_ps(ZZ7_real, ZZ7_real), _mm_mul_ps(ZZ7_imag, ZZ7_imag)));
        ZZZ_real = _mm_mul_ps((_mm_add_ps(_mm_mul_ps(Z6_real, ZZ7_real), _mm_mul_ps(Z6_imag, ZZ7_imag))), division_factor); 

        _mm_stream_ps(&voigt_value[i], ZZZ_real);
    }
    _mm_free(stmp);
}
#elif defined __ALTIVEC__
/* a vectorized version of the Voigt function using Altivec / VMX instructions */
void my_voigt(const float *damping, const float *frequency_offset, float *voigt_value, int N)
{
   // coefficients of the rational approximation formula
   // to the complementary error function
   const vector float A0 = (vector float) (122.607931777104326f);
   const vector float A1 = (vector float) (214.382388694706425f);
   const vector float A2 = (vector float) (181.928533092181549f);
   const vector float A3 = (vector float) (93.155580458138441f);
   const vector float A4 = (vector float) (30.180142196210589f);
   const vector float A5 = (vector float) (5.912626209773153f);
   const vector float A6 = (vector float) (0.564189583562615f);
   const vector float B0 = (vector float) (122.60793177387535f);
   const vector float B1 = (vector float) (352.730625110963558f);
   const vector float B2 = (vector float) (457.334478783897737f);
   const vector float B3 = (vector float) (348.703917719495792f);
   const vector float B4 = (vector float) (170.354001821091472f);
   const vector float B5 = (vector float) (53.992906912940207f);
   const vector float B6 = (vector float) (10.479857114260399f);

   vector float ivsigno;
   vector float V;
   vector float Z1_real;
   vector float Z1_imag;
   vector float Z2_real;
   vector float Z2_imag;
   vector float Z3_real;
   vector float Z3_imag;
   vector float Z4_real;
   vector float Z4_imag;
   vector float Z5_real;
   vector float Z5_imag;
   vector float Z6_real;
   vector float Z6_imag;
   vector float ZZ1_real;
   vector float ZZ1_imag;
   vector float ZZ2_real;
   vector float ZZ2_imag;
   vector float ZZ3_real;
   vector float ZZ3_imag;
   vector float ZZ4_real;
   vector float ZZ4_imag;
   vector float ZZ5_real;
   vector float ZZ5_imag;
   vector float ZZ6_real;
   vector float ZZ6_imag;
   vector float ZZ7_real;
   vector float ZZ7_imag;
   vector float division_factor;
   vector float ZZZ_real;

   vector bool int mask;
   const vector float one = (vector float) (1.0f);
   const vector float zero = (vector float) (-0.0f);
   const vector float mone = (vector float) (-1.0f);

   vector float damp;
   vector float offs;

   for(int i=0; i<N; i+=4){
      damp = vec_ld(0,(float *) &damping[i]);
      offs = vec_ld(0,(float *) &frequency_offset[i]);
      mask = vec_cmplt(offs,zero);
      ivsigno = vec_sel(mone, one, mask);
      //ivsigno = (vector float) (1.0f);
      V = vec_madd(ivsigno, offs, zero);
      Z1_real = vec_madd(A6, damp, A5);
      Z1_imag = vec_nmsub(A6, V, zero);
      Z2_real = vec_add(vec_madd(Z1_real,damp,zero),vec_madd(Z1_imag,V,A4)); 
      Z2_imag = vec_add(vec_nmsub(Z1_real,V,zero),vec_madd(Z1_imag,damp,zero));
      Z3_real = vec_add(vec_madd(Z2_real,damp,zero),vec_madd(Z2_imag,V,A3));
      Z3_imag = vec_add(vec_nmsub(Z2_real,V,zero),vec_madd(Z2_imag,damp,zero));
      Z4_real = vec_add(vec_madd(Z3_real,damp,zero),vec_madd(Z3_imag,V,A2));
      Z4_imag = vec_add(vec_nmsub(Z3_real,V,zero),vec_madd(Z3_imag,damp,zero));
      Z5_real = vec_add(vec_madd(Z4_real,damp,zero),vec_madd(Z4_imag,V,A1));
      Z5_imag = vec_add(vec_nmsub(Z4_real,V,zero),vec_madd(Z4_imag,damp,zero));
      Z6_real = vec_add(vec_madd(Z5_real,damp,zero),vec_madd(Z5_imag,V,A0));
      Z6_imag = vec_add(vec_nmsub(Z5_real,V,zero),vec_madd(Z5_imag,damp,zero));
      ZZ1_real = vec_add(damp,B6); 
      ZZ1_imag = vec_madd(mone,V,zero);
      ZZ2_real = vec_add(vec_madd(ZZ1_real,damp,zero),vec_madd(ZZ1_imag,V,B5));
      ZZ2_imag = vec_add(vec_nmsub(ZZ1_real,V,zero),vec_madd(ZZ1_imag,damp,zero));
      ZZ3_real = vec_add(vec_madd(ZZ2_real,damp,zero),vec_madd(ZZ2_imag,V,B4));
      ZZ3_imag = vec_add(vec_nmsub(ZZ2_real,V,zero),vec_madd(ZZ2_imag,damp,zero));
      ZZ4_real = vec_add(vec_madd(ZZ3_real,damp,zero),vec_madd(ZZ3_imag,V,B3));
      ZZ4_imag = vec_add(vec_nmsub(ZZ3_real,V,zero),vec_madd(ZZ3_imag,damp,zero));
      ZZ5_real = vec_add(vec_madd(ZZ4_real,damp,zero),vec_madd(ZZ4_imag,V,B2));
      ZZ5_imag = vec_add(vec_nmsub(ZZ4_real,V,zero),vec_madd(ZZ4_imag,damp,zero));
      ZZ6_real = vec_add(vec_madd(ZZ5_real,damp,zero),vec_madd(ZZ5_imag,V,B1));
      ZZ6_imag = vec_add(vec_nmsub(ZZ5_real,V,zero),vec_madd(ZZ5_imag,damp,zero));
      ZZ7_real = vec_add(vec_madd(ZZ6_real,damp,zero),vec_madd(ZZ6_imag,V,B0));
      ZZ7_imag = vec_add(vec_nmsub(ZZ6_real,V,zero),vec_madd(ZZ6_imag,damp,zero));
      division_factor = vec_div(one,vec_madd(ZZ7_real,ZZ7_real,vec_madd(ZZ7_imag,ZZ7_imag,zero)));
      ZZZ_real = vec_madd(vec_madd(Z6_real,ZZ7_real,vec_madd(Z6_imag,ZZ7_imag,zero)),division_factor,zero); 
      vec_st(ZZZ_real,0,(float *)&voigt_value[i]);
   }
}

/* An accurate vector division routine using the reciprocal estimate and 
 * two Newton-Raphson iterations
 */
static inline vector float vec_div(vector float A, vector float B)
{
     vector float y0;
     vector float y1;
     vector float y2;
     vector float Q;
     vector float R;
     vector float one = (vector float) (1.0f);
     vector float zero = (vector float) (-0.0f);
     vector float mone = (vector float) (-1.0f);

     y0 = vec_re(B);            // approximate 1/B

     // y1 = y0*(-(y0*B - 1.0))+y0  i.e. y0+y0*(1.0 - y0*B)
     y1 = vec_madd(y0,vec_nmsub(y0, B, one),y0);
  
     // REPEAT the Newton-Raphson to get the required 24 bits
     y2 = vec_madd(y1, vec_nmsub(y1, B, one),y1);

     // y2 = y1*(-(y1*B - 1.0f))+y1  i.e. y1+y1*(1.0f - y1*B)
     // y2 is now the correctly rounded reciprocal, and the manual considers this
     // OK for use in computing the remainder: Q = A*y2, R = A - B*Q

     Q = vec_madd(A,y2,zero);  // -0.0 IEEE
     R = vec_nmsub(B,Q,A);      // -(B*Q-A) == (A-B*Q)

     // final rouding adjustment
     return(vec_madd(R, y2, Q));
}

#else
/* a non-vectorized microprocessor version */
void my_voigt(const float *damping, const float *frequency_offset, float *voigt_value, int N)
{                                                                  
    // coefficients of the rational approximation formula
    // to the complementary error function
    const float A0 = 122.607931777104326f;
    const float A1 = 214.382388694706425f;
    const float A2 = 181.928533092181549f;
    const float A3 = 93.155580458138441f;
    const float A4 = 30.180142196210589f;
    const float A5 = 5.912626209773153f;
    const float A6 = 0.564189583562615f;
    const float B0 = 122.60793177387535f;
    const float B1 = 352.730625110963558f;
    const float B2 = 457.334478783897737f;
    const float B3 = 348.703917719495792f;
    const float B4 = 170.354001821091472f;
    const float B5 = 53.992906912940207f;
    const float B6 = 10.479857114260399f;

    float ivsigno;
    float V;
    float Z1_real;
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

    register float damp;
    float offs;

    int i;
    for(i=0; i<N; ++i){
       damp = damping[i];
       offs = frequency_offset[i];
       if (offs < 0)
           ivsigno=-1;
       else
           ivsigno=1;
       V = ivsigno * offs;
       Z1_real = A6 * damp + A5;
       Z1_imag = A6 * -V;
       Z2_real = Z1_real * damp - Z1_imag * -V        + A4;
       Z2_imag = Z1_real * -V      + Z1_imag * damp;
       Z3_real = Z2_real * damp - Z2_imag * -V        + A3;
       Z3_imag = Z2_real * -V      + Z2_imag * damp;
       Z4_real = Z3_real * damp - Z3_imag * -V        + A2;
       Z4_imag = Z3_real * -V      + Z3_imag * damp;
       Z5_real = Z4_real * damp - Z4_imag * -V        + A1;
       Z5_imag = Z4_real * -V      + Z4_imag * damp;
       Z6_real = Z5_real * damp - Z5_imag * -V        + A0;
       Z6_imag = Z5_real * -V      + Z5_imag * damp;
       ZZ1_real = damp + B6;
       ZZ1_imag = -V;
       ZZ2_real = ZZ1_real * damp - ZZ1_imag * -V     + B5;
       ZZ2_imag = ZZ1_real * -V      + ZZ1_imag * damp;
       ZZ3_real = ZZ2_real * damp - ZZ2_imag * -V     + B4;
       ZZ3_imag = ZZ2_real * -V      + ZZ2_imag * damp;
       ZZ4_real = ZZ3_real * damp - ZZ3_imag * -V     + B3;
       ZZ4_imag = ZZ3_real * -V      + ZZ3_imag * damp;
       ZZ5_real = ZZ4_real * damp - ZZ4_imag * -V     + B2;
       ZZ5_imag = ZZ4_real * -V      + ZZ4_imag * damp;
       ZZ6_real = ZZ5_real * damp  - ZZ5_imag * -V    + B1;
       ZZ6_imag = ZZ5_real * -V       + ZZ5_imag * damp;
       ZZ7_real = ZZ6_real * damp  - ZZ6_imag * -V    + B0;
       ZZ7_imag = ZZ6_real * -V       + ZZ6_imag * damp;
       division_factor = 1.0f / (ZZ7_real * ZZ7_real + ZZ7_imag * ZZ7_imag);
       ZZZ_real = (Z6_real * ZZ7_real  + Z6_imag * ZZ7_imag) * division_factor;
       voigt_value[i]=ZZZ_real;
   }

}

#endif
