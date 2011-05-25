NVCC      = nvcc
CC        = cc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG
LDFLAGS   = -O2 -m64
OBJ       = main.o dot_prod.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

dot_prod: $(OBJ)
	$(NVCC) -o dot_prod $(LDFLAGS) $(OBJ)

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f dot_prod $(OBJ)
