NVCC      = nvcc
CC        = cc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG
LDFLAGS   = -O2 -m64
OBJ       = main.o histogram.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

histogram: $(OBJ) 
	$(NVCC) -o histogram $(LDFLAGS) $(OBJ)

%.o: %.c 
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f histogram $(OBJ)
