CC  = nvcc -O3
TARGET  = a.out
SRCS    = mat_mult_cuda.cu
OBJS    = $(SRCS:.cu=.o)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

all: clean $(OBJS) $(TARGET)

run: ${TARGET}
	./${TARGET}

clean:
	-rm -f *.o $(TARGET)

bench: ${TARGET}
	./bench.sh > result.txt