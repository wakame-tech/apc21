CC  = nvcc
TARGET  = a.out
SRCS    = mat_mult_cuda.cu
OBJS    = $(SRCS:.cu=.o)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

all: clean $(OBJS) $(TARGET)

run:
	./${TARGET}

clean:
	-rm -f $(OBJS) $(TARGET) *.d

bench:
	make && ./bench.sh > result.txt