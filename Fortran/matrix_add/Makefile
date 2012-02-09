FC        = pgfortran
FFLAGS    = -O2 -DDEBUG
CUDAFLAGS = $(FFLAGS) -Minfo=accel -Mcuda -ta=nvidia,cuda3.2
LDFLAGS   = -Mcuda
OBJ       = main.o

.SUFFIXES:
.SUFFIXES: .o .for .cuf

mat_add: $(OBJ)
	$(FC) -o mat_add $(LDFLAGS) $(OBJ)

%.o: %.for
	$(FC) -c $(FFLAGS) $< 

%.o: %.cuf
	$(FC) -c $(CUDAFLAGS) $<

clean:
	rm -f mat_add $(OBJ) *.mod *.out
