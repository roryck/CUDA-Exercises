#ifndef Voigt_h_
#define Voigt_h_ 1

#include <math.h>

void my_voigt(const float *damping, const float *frequency_offset, float *voigt_value, int N);

#endif // Voigt_h_
