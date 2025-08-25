CC = hipcc
CFLAGS = -O3 -I.. -w -std=c++20

all:
	@echo "Please specify a target to build"
%.out: %.cc
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f *.out

.PHONY: all clean