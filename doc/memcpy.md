# Memcpy 

This sample illustrates:

- the *code conventions* of this repository
- the basic usage of `Profiler` in `tools/helper.hh` to measure kernel performance 
    - `add` method to add a kernel to be profiled, including `name`, `kernel lambda`, `validate` function, `bytes` processed, and `flops` performed
    - `runAll` method to execute all added kernels and display their performance
- the performance of `hipMemcpy` for different memory copy types: `H2D`, `D2H`, `D2D`

Codes available in [memcpy.cc](../examples/memcpy.cc)

To build and run this sample:

```bash
# in examples/ directory
make memcpy.out
./memcpy.out <num_elements> <timed_runs> <warmup_runs>
```

Insights:
- `hipMemcpy` performance varies based on the type of memory copy operation (H2D, D2H, D2D).
- `H2D` and `D2H` transfers typically use the PCIe bus
- `D2D` transfer speed depends on hardware's peak device memory bandwidth(within the same GPU)
