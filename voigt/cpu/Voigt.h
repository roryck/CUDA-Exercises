#ifndef Voigt_h_
#define Voigt_h_ 1

#include <math.h>

#define  my_voigt(damping,frequency_offset,voigt_value)             \
{                                                                   \
    if (frequency_offset < 0)                                       \
        CSAC::VOIGT::ivsigno=-1;                                    \
    else                                                            \
        CSAC::VOIGT::ivsigno=1;                                     \
    CSAC::VOIGT::V = CSAC::VOIGT::ivsigno * frequency_offset;       \
    CSAC::VOIGT::Z1_real = CSAC::VOIGT::A6 * damping + CSAC::VOIGT::A5; \
    CSAC::VOIGT::Z1_imag = CSAC::VOIGT::A6 * -CSAC::VOIGT::V;   \
    CSAC::VOIGT::Z2_real = CSAC::VOIGT::Z1_real * damping - CSAC::VOIGT::Z1_imag * -CSAC::VOIGT::V        + CSAC::VOIGT::A4; \
    CSAC::VOIGT::Z2_imag = CSAC::VOIGT::Z1_real * -CSAC::VOIGT::V      + CSAC::VOIGT::Z1_imag * damping; \
    CSAC::VOIGT::Z3_real = CSAC::VOIGT::Z2_real * damping - CSAC::VOIGT::Z2_imag * -CSAC::VOIGT::V        + CSAC::VOIGT::A3; \
    CSAC::VOIGT::Z3_imag = CSAC::VOIGT::Z2_real * -CSAC::VOIGT::V      + CSAC::VOIGT::Z2_imag * damping; \
    CSAC::VOIGT::Z4_real = CSAC::VOIGT::Z3_real * damping - CSAC::VOIGT::Z3_imag * -CSAC::VOIGT::V        + CSAC::VOIGT::A2; \
    CSAC::VOIGT::Z4_imag = CSAC::VOIGT::Z3_real * -CSAC::VOIGT::V      + CSAC::VOIGT::Z3_imag * damping; \
    CSAC::VOIGT::Z5_real = CSAC::VOIGT::Z4_real * damping - CSAC::VOIGT::Z4_imag * -CSAC::VOIGT::V        + CSAC::VOIGT::A1; \
    CSAC::VOIGT::Z5_imag = CSAC::VOIGT::Z4_real * -CSAC::VOIGT::V      + CSAC::VOIGT::Z4_imag * damping; \
    CSAC::VOIGT::Z6_real = CSAC::VOIGT::Z5_real * damping - CSAC::VOIGT::Z5_imag * -CSAC::VOIGT::V        + CSAC::VOIGT::A0; \
    CSAC::VOIGT::Z6_imag = CSAC::VOIGT::Z5_real * -CSAC::VOIGT::V      + CSAC::VOIGT::Z5_imag * damping; \
    CSAC::VOIGT::ZZ1_real = damping + CSAC::VOIGT::B6;          \
    CSAC::VOIGT::ZZ1_imag = -CSAC::VOIGT::V;                    \
    CSAC::VOIGT::ZZ2_real = CSAC::VOIGT::ZZ1_real * damping - CSAC::VOIGT::ZZ1_imag * -CSAC::VOIGT::V     + CSAC::VOIGT::B5; \
    CSAC::VOIGT::ZZ2_imag = CSAC::VOIGT::ZZ1_real * -CSAC::VOIGT::V      + CSAC::VOIGT::ZZ1_imag * damping; \
    CSAC::VOIGT::ZZ3_real = CSAC::VOIGT::ZZ2_real * damping - CSAC::VOIGT::ZZ2_imag * -CSAC::VOIGT::V     + CSAC::VOIGT::B4; \
    CSAC::VOIGT::ZZ3_imag = CSAC::VOIGT::ZZ2_real * -CSAC::VOIGT::V      + CSAC::VOIGT::ZZ2_imag * damping; \
    CSAC::VOIGT::ZZ4_real = CSAC::VOIGT::ZZ3_real * damping - CSAC::VOIGT::ZZ3_imag * -CSAC::VOIGT::V     + CSAC::VOIGT::B3; \
    CSAC::VOIGT::ZZ4_imag = CSAC::VOIGT::ZZ3_real * -CSAC::VOIGT::V      + CSAC::VOIGT::ZZ3_imag * damping; \
    CSAC::VOIGT::ZZ5_real = CSAC::VOIGT::ZZ4_real * damping - CSAC::VOIGT::ZZ4_imag * -CSAC::VOIGT::V     + CSAC::VOIGT::B2; \
    CSAC::VOIGT::ZZ5_imag = CSAC::VOIGT::ZZ4_real * -CSAC::VOIGT::V      + CSAC::VOIGT::ZZ4_imag * damping; \
    CSAC::VOIGT::ZZ6_real = CSAC::VOIGT::ZZ5_real * damping  - CSAC::VOIGT::ZZ5_imag * -CSAC::VOIGT::V    + CSAC::VOIGT::B1; \
    CSAC::VOIGT::ZZ6_imag = CSAC::VOIGT::ZZ5_real * -CSAC::VOIGT::V       + CSAC::VOIGT::ZZ5_imag * damping; \
    CSAC::VOIGT::ZZ7_real = CSAC::VOIGT::ZZ6_real * damping  - CSAC::VOIGT::ZZ6_imag * -CSAC::VOIGT::V    + CSAC::VOIGT::B0; \
    CSAC::VOIGT::ZZ7_imag = CSAC::VOIGT::ZZ6_real * -CSAC::VOIGT::V       + CSAC::VOIGT::ZZ6_imag * damping; \
    CSAC::VOIGT::division_factor = 1.0f / (CSAC::VOIGT::ZZ7_real * CSAC::VOIGT::ZZ7_real + CSAC::VOIGT::ZZ7_imag * CSAC::VOIGT::ZZ7_imag); \
    CSAC::VOIGT::ZZZ_real =                                     \
    (CSAC::VOIGT::Z6_real * CSAC::VOIGT::ZZ7_real  + CSAC::VOIGT::Z6_imag * CSAC::VOIGT::ZZ7_imag) * \
    CSAC::VOIGT::division_factor;                               \
    CSAC::VOIGT::ZZZ_imag =                                     \
    (CSAC::VOIGT::Z6_real * -CSAC::VOIGT::ZZ7_imag + CSAC::VOIGT::Z6_imag * CSAC::VOIGT::ZZ7_real) * \
    CSAC::VOIGT::division_factor;                               \
    voigt_value=CSAC::VOIGT::ZZZ_real;                          \
}
#endif // Voigt_h_
