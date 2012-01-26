# set compiler to desired compiler (gcc, intel, xlc)
# set problem_size to desired problem size (tiny, small, medium, large)
# set openmp to enable / disable OpenMP (true, false)
# set vector to enable use of SIMD code versions (sse, altivec, none)

compiler     = intel
problem_size = small
openmp       = true
vector       = sse

# Compiler section
ifeq ($(compiler),intel)
CC = icc
CXX = icpc
CFLAGS = -O3 -xT
LDFLAGS = 
else ifeq ($(compiler),gcc) 
CC = gcc
CXX = g++
CFLAGS = -O2 
LDFLAGS = 
else ifeq ($(compiler),xlc)
CC = xlc
CXX = xlC
CFLAGS = -O3 
LDFLAGS = 
endif

ifeq ($(problem_size),tiny)
SIZE = -DPROBLEM_SIZE=8
else ifeq ($(problem_size),small)
SIZE = -DPROBLEM_SIZE=4096
else ifeq ($(problem_size),medium)
SIZE = -DPROBLEM_SIZE=8192
else ifeq ($(problem_size),large)
SIZE = -DPROBLEM_SIZE=16384
endif

ifeq ($(openmp),true)
ifeq ($(compiler),intel)
CFLAGS += -openmp
LDFLAGS += -openmp
else ifeq ($(compiler),xlc)
CFLAGS += -qthreaded -qsmp=omp
LDFLAGS += -qthreaded -qsmp=omp
endif
endif

ifeq ($(vector),sse)
CFLAGS += -D_SSE
else ifeq ($(vector),altivec)
CFLAGS += -qenablevmx -qarch=ppc64v -qaltivec
endif

objects = subtract_timeval.o single.o ransign.o Voigt.o

test_voigt: Voigt.h $(objects)
	$(CXX) -o test_voigt $(LDFLAGS) $(objects)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $(SIZE) $(TIMING) $<

%.o: %.c 
	$(CC) -c $(CFLAGS) $(SIZE) $(TIMING) $<

clean:
	rm -f test_voigt $(objects) $(chkobjs)
