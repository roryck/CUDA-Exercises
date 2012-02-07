FC        = pgfortran
FFLAGS    = -O2 -DDEBUG
CUDAFLAGS = $(FFLAGS) -Mcuda
LDFLAGS   = -Mcuda
OBJ       = vec_add_mod.o main.o

.SUFFIXES:
.SUFFIXES: .o .for .cuf

vec_add: $(OBJ)
	$(FC) -o vec_add $(LDFLAGS) $(OBJ)

%.o: %.for
	$(FC) -c $(FFLAGS) $< 

%.o: %.cuf
	$(FC) -c $(CUDAFLAGS) $<

clean:
	rm -f vec_add $(OBJ) *.mod *.out
