# <https://zenn.dev/yagiyuki/articles/b5545c3b546bbeb662bf>
CC  = g++ -fopenmp -O3 -std=c++11
TARGET  = a.out
SRCS    = mat_mult_omp.cpp
OBJS    = $(SRCS:.cpp=.o)

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