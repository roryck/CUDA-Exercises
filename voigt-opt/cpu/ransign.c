#define MAX( x, y ) ((x) >= (y) ? (x) : (y))
float ransign(long int *idum){
   /* local variables */
   static long int idum2=123456789;
   static long int iy=0;
   static long int iv[32] = {0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0}; 
   long int im1, im2, imm1, imh, ia1, ia2, iq1, iq2, ir1, ir2, ntab, ndiv;
   long int j, k;
   float ret;
  
   /* set values */
   im1 = 2147483563;
   im2 = 2147483399;
   imm1 = im1 - 1;
   imh = imm1/2;
   
   ia1 = 40014;
   ia2 = 40692;
   iq1 = 53668;
   iq2 = 52774;
   ir1 = 12211;
   ir2 = 3791;
   ntab = 32;
   ndiv = 1 + imm1 / ntab;

   if(*idum <= 0){
       *idum = MAX(-(*idum), 1);
       idum2 = *idum;
       for(j=ntab+7; j>=0; j--){
          k = *idum / iq1;
          *idum = ia1*(*idum - k*iq1) - k*ir1;
          if(*idum < 0) *idum = *idum + im1;
          if(j < ntab) iv[j] = *idum;
       }
       iy = iv[0];
    }
    k = *idum / iq1;
    *idum = ia1 * (*idum - k*iq1) - k*ir1;
    if (*idum < 0) *idum = *idum + im1;
    k = idum2 / iq2;
    idum2 = ia2 * (idum2 - k*iq2) - k*ir2;
    if(idum2 < 0) idum2 = idum2 + im2;
    j = 0 + iy / ndiv;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1) iy = iy + imm1;
    ret = (iy < imh ? -1.0f : 1.0); 
    return(ret);
}
