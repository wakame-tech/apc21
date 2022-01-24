# <https://zenn.dev/yagiyuki/articles/b5545c3b546bbeb662bf>
CC  = mpic++
TARGET  = a.out
SRCS    = mat_mult_mpi.cpp
OBJS    = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

all: clean $(OBJS) $(TARGET)

run:
	mpiexec -n 4 ./${TARGET}

clean:
	-rm -f $(OBJS) $(TARGET) *.d
