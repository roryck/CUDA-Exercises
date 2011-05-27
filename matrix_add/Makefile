NVCC      = nvcc
CC        = nvcc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG -m64
LDFLAGS   = -O2 -m64
OBJ       = main.o array.o double.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

test: $(OBJ)
	$(NVCC) -o test $(LDFLAGS) $(OBJ)

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f test $(OBJ)
