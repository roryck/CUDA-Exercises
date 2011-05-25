NVCC      = nvcc
CC        = cc
NVCCFLAGS = -O2 -DDEBUG -m64
CFLAGS    = -O2 -DDEBUG
LDFLAGS   = -O2 -m64
OBJ       = main.o vec_add.o

.SUFFIXES:
.SUFFIXES: .o .c .cu

vec_add: $(OBJ)
	$(NVCC) -o vec_add $(LDFLAGS) $(OBJ)

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $<

clean:
	rm -f vec_add $(OBJ)
