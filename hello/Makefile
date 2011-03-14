NVCC      = nvcc
CC        = cc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG
LDFLAGS   = -O2 -m64
OBJ       = main.o hello.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

hello: $(OBJ)
	$(NVCC) -o hello $(LDFLAGS) $(OBJ)

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f hello $(OBJ)
