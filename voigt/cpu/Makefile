# set the values for compiler, timers and problem_size
# set compiler to desired compiler (gcc, intel, xlc)
# set timers to desired timing method (none, user, wall, both) 
# set problem_size to desired problem size (small, medium, large)
compiler     = intel
timers       = both
problem_size = small
 
ifeq ($(compiler),intel)
CC = icc
CXX = icpc
CFLAGS = -O2 -ip -xT
LDFLAGS =
else ifeq ($(compiler),gcc) 
CC = gcc
CXX = g++
CFLAGS = -O2
LDFLAGS =
else ifeq ($(compiler),xlc)
CC = xlc
CXX = xlC
CFLAGS = -O2
LDFLAGS =
endif

ifeq ($(timers),none)
TIMING =
else ifeq ($(timers),user)
TIMING = -D_VOIGT_USERTIME_
else ifeq ($(timers),wall)
TIMING = -D_VOIGT_WALLTIME_
else ifeq ($(timers),both)
TIMING = -D_VOIGT_USERTIME_ -D_VOIGT_WALLTIME_
endif
ifeq ($(problem_size),small)
SIZE = -DPROBLEM_SIZE=4096
else ifeq ($(problem_size),medium)
SIZE = -DPROBLEM_SIZE=8192
else ifeq ($(problem_size),large)
SIZE = -DPROBLEM_SIZE=16384
endif

objects = subtract_timeval.o single.o

test_voigt: Voigt.h $(objects)
	$(CXX) -o test_voigt $(LDFLAGS) $(objects)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $(SIZE) $(TIMING) $<

%.o: %.c 
	$(CC) -c $(CFLAGS) $(SIZE) $(TIMING) $<

clean:
	rm -f test_voigt $(objects) $(chkobjs)
