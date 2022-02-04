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

bench_n: ${TARGET}
	./bench_n.sh > result_n.txt

bench_pn: ${TARGET}
	./bench_pn.sh > result_pn.txt