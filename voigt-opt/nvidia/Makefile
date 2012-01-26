# set the values for compiler, timers and problem_size
# set compiler to desired compiler (gcc, intel, xlc)
# set timers to desired timing method (none, user, wall, both) 
# set problem_size to desired problem size (small, medium, large)
debug        = false
emulate      = false
problem_size = small
timing       = prof
openmp       = false
 
CC = nvcc
CXX = nvcc
NVCC = nvcc
CFLAGS = -O2 -m64
NVCCFLAGS = -O2 -m64 -I/Developer/CUDA/common/inc
LDFLAGS = -m64

ifeq ($(debug),true)
NVCCFLAGS += -g -D_DEBUG
endif
ifeq ($(emulate),true)
NVCCFLAGS += -deviceemu
LDFLAGS   += -deviceemu
endif
ifeq ($(problem_size),tiny)
SIZE = -DPROBLEM_SIZE=512
else ifeq ($(problem_size),small)
SIZE = -DPROBLEM_SIZE=4096
else ifeq ($(problem_size),medium)
SIZE = -DPROBLEM_SIZE=8192
else ifeq ($(problem_size),large)
SIZE = -DPROBLEM_SIZE=16384
endif
ifeq ($(timing),prof)
NVCCFLAGS += -D_TIMING_PROF_
endif
ifeq ($(openmp),true)
NVCCFLAGS += -Xcompiler -fopenmp
LDFLAGS += -lgomp
endif

objects = single.o voigt.o ransign.o

test_voigt: voigt.h $(objects)
	$(NVCC) -o test_voigt $(LDFLAGS) $(objects)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $(SIZE) $(TIMING) $<

%.o: %.c 
	$(CC) -c $(CFLAGS) $(SIZE) $(TIMING) $<

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(SIZE) $(TIMING) $<

clean:
	rm -f test_voigt $(objects) $(chkobjs)
