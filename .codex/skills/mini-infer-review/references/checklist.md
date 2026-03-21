# Mini Infer Review Checklist

## Correctness

- [ ] New API calls use the right field names and parameters
- [ ] `dry_run=True` paths are not accidentally broken
- [ ] KV cache allocations are freed on all relevant paths
- [ ] Multi-device code passes device arguments explicitly and does not hide cross-device copies

## Performance hot path

- [ ] Per-layer hooks or patched forwards do not call `.item()`, `.numpy()`, `bool(cuda_tensor)`, `int(cuda_tensor)`, or `print(cuda_tensor)`
- [ ] New code does not add hidden CPU-GPU sync in decode hot paths

## Test coverage gaps

- [ ] New benchmark or profiling scripts can at least construct config in dry-run mode
- [ ] GPU-only paths are called out as "real GPU validation required" when no unit coverage exists

## Benchmark methodology

- [ ] Workload and environment are explicit
- [ ] Reported numbers come from real runs, not estimates
- [ ] `TTFT`, `TPOT`, and `peak memory` follow the benchmark metrics spec

## Documentation

- [ ] File headers or top-level explanations stay accurate when behavior changes
- [ ] README or other calling docs stay in sync when public behavior changes
