NVCC      = nvcc
CC        = nvcc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG -m64
LDFLAGS   = -O2 -m64
OBJ       = main.o array.o matrix_add.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

matrix_add: $(OBJ)
	$(NVCC) -o matrix_add $(LDFLAGS) $(OBJ)

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f matrix_add $(OBJ)
