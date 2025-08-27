CC = hipcc
CFLAGS = -O3 -w -std=c++20
LDFLAGS = -lrocsparse -lrocblas
INCLUDE = -I.. 
all:
	@echo "Please specify a target to build"
%.out: %.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDE) -o $@ $<
clean:
	rm -f *.out

.PHONY: all clean