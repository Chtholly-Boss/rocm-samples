# Reduction

This sample illustrates:

- basic implementation of reduction $out = sum(in)$ operation on GPU, where `in` is a vector and `out` is a scalar
- the impact of **Vectorization** on performance
- different implementation of `warp_reduce_sum` using `DPP instructions` and `shuffle` instructions

Codes available in [reduce.cc](../examples/reduce.cc)

Insights:
- Reduction is a memory-bound operation, and its performance is limited by memory bandwidth
- Vectorization can improve memory throughput by allowing multiple data elements to be processed in a single memory transaction
- Using `DPP instructions` for warp-level reduction is slightly faster than using `shuffle` instructions, but the difference is not significant
