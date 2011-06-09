#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#include <assert.h>
#include <limits.h>
#include <float.h>

//#include "netcdf.h"
#include "Voigt.h"

// The standard test case sizes
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 16384
#endif

// Subtract the value of time timeval structure for resource usage.
extern "C"
{
    int timeval_subtract (struct timeval *result, struct timeval*x, struct timeval*y);
}

using namespace std;

namespace CSAC
{
    namespace VOIGT
    {
        /* coefficients of the rational approximation formula
           to the complementary error function 
        */ 
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
        const float XDWS[]={
            .1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f,.9f,1.0f,1.2f,1.4f,1.6f,1.8f,2.0f,   
            3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,12.0f,14.0f,16.0f,18.0f,20.0f
        };
        const float YDWS[]={   
            9.9335991E-02f,1.9475104E-01f,2.8263167E-01f,3.5994348E-01f,  
            4.2443639E-01f,4.7476321E-01f,5.1050407E-01f,5.3210169E-01f,  
            5.4072434E-01f,5.3807950E-01f,5.0727350E-01f,4.5650724E-01f,  
            3.9993989E-01f,3.4677279E-01f,3.0134040E-01f,1.7827103E-01f,  
            1.2934799E-01f,1.0213407E-01f,8.4542692E-02f,7.2180972E-02f,  
            6.3000202E-02f,5.5905048E-02f,5.0253846E-02f,4.1812878E-02f,  
            3.5806101E-02f,3.1311397E-02f,2.7820844E-02f,2.5031367E-02f
        };
    
        int i;
        int k;
        int kk; 
        int kkk;
        int ivsigno;
        float V;
        float Y;
        float D;
        float D1;
        float D2;
        float D3;
        float D12; 
        float D13; 
        float D23;
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
        float ZZZ_imag;
    
    }; // end namespace CSAC::voigt
} // end namespace CSAC


int main(int argc, char* argv[])
{
    char *program_name_ = argv[0];

    // Timers
#ifdef _VOIGT_USERTIME_
    struct rusage usage1;
    struct rusage usage2;
#endif 
#ifdef _VOIGT_WALLTIME_
    struct timeval start;
    struct timeval stop;
#endif
    struct timeval result;

    int i=0;
    int j=0;

    float voigt_value, faraday_value;
    float vals;
    float a,b;
    float step = 32.0/PROBLEM_SIZE;

    assert(FLT_EPSILON < step);

    vector <float> row;
    row.resize(PROBLEM_SIZE);

    // create vectors to hold input and output data
    vector< vector<float> > damping;
    damping.resize(PROBLEM_SIZE);
    for (int ii = 0; ii < PROBLEM_SIZE; ii++)
        damping[ii] = row;

    vector< vector<float> > offset;
    offset.resize(PROBLEM_SIZE);
    for (int ii = 0; ii < PROBLEM_SIZE; ii++)
        offset[ii] = row;

    vector< vector<float> > voigt;
    voigt.resize(PROBLEM_SIZE);
    for (int ii = 0; ii < PROBLEM_SIZE; ii++)
        voigt[ii] = row; 

    /* assign initial data values */
    vals = 10.0;
    for(i=0; i<PROBLEM_SIZE; ++i){
        vals += step;
        for(j=0; j<PROBLEM_SIZE; ++j){
            damping[i][j] = vals;
            offset[j][i] = vals;
        }
    }
    
    // Get timer for usage so far, notice that we have done nothing
#ifdef _VOIGT_USERTIME_
    if (getrusage(RUSAGE_SELF, &usage1) != 0)
    {
        cerr<<program_name_<<": getrusage failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }
#endif
#ifdef _VOIGT_WALLTIME_
    if (gettimeofday(&start, NULL) != 0)
    {
        cerr<<program_name_<<": gettimeofday failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }
#endif

    /*** ----  main compute kernel ----- ***/
    /*** this is where the magic happens ***/
    for(i=0; i<PROBLEM_SIZE; ++i){
        for(j=0; j<PROBLEM_SIZE; ++j){
          my_voigt(damping[i][j], offset[i][j], voigt[i][j], faraday_value);
        }
    }

#ifdef _VOIGT_USERTIME_
    if (getrusage(RUSAGE_SELF, &usage2) != 0)
    {
        cerr<<"getrusage failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }
    timeval_subtract (&result, &usage2.ru_utime, &usage1.ru_utime);
    cout<<program_name_<<": total user time: "<<result.tv_sec<<" seconds and "<<result.tv_usec<<" microseconds"<<endl;
#endif
#ifdef _VOIGT_WALLTIME_
    if (gettimeofday(&stop, NULL) != 0)
    {
        cerr<<"getrusage failed, it returns: "<<strerror(errno)<<endl;
        return 1;
    }
    timeval_subtract(&result, &stop, &start);
    cout<<program_name_<<": total wallclock time: "<<result.tv_sec<<" seconds and "<<result.tv_usec<<" microseconds"<<endl;
#endif

}

