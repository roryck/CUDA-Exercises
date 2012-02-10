FC        = pgfortran
FFLAGS    = -O2 -DDEBUG
CUDAFLAGS = $(FFLAGS) -Mcuda
LDFLAGS   = -Mcuda
OBJ       = vec_rev_mod.o main.o

.SUFFIXES:
.SUFFIXES: .o .for .cuf

vec_rev: $(OBJ)
	$(FC) -o vec_rev $(LDFLAGS) $(OBJ)

%.o: %.for
	$(FC) -c $(FFLAGS) $< 

%.o: %.cuf
	$(FC) -c $(CUDAFLAGS) $<

clean:
	rm -f vec_rev $(OBJ) *.mod *.out
