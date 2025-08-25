# Y = aX

This sample illustrates:

- basic implementation of $y = a x$ operation on GPU, where `a` is a scalar and `x` and `y` are vectors
- the impact of **Vectorization** on performance
- the impact of **Occupancy** on performance

Codes available in [ax.cc](../examples/ax.cc)

To build and run this sample:

```bash
# in examples/ directory
make ax.out
./ax.out <num_elements> <timed_runs> <warmup_runs>
```

Insights:
- `Y = aX` is a memory-bound operation, and its performance is limited by memory bandwidth
- Vectorization can improve memory throughput by allowing multiple data elements to be processed in a single memory transaction
- Higher occupancy can help hide memory latency, but beyond a certain point, increasing occupancy yields diminishing returns
