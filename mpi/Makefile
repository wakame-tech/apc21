CC  = mpic++ -O3
TARGET  = a.out
SRCS    = mat_mult_mpi.cpp
OBJS    = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

all: clean $(OBJS) $(TARGET)

run: $(TARGET)
	mpiexec -n ${p} ./${TARGET} ${n}

clean:
	-rm -f *.o $(TARGET)

bench_p: ${TARGET}
	./bench_p.sh > result_p.txt

bench_pn: ${TARGET}
	./bench_pn.sh > result_pn.txt