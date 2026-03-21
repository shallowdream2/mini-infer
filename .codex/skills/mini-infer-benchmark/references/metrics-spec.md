# Mini Infer Benchmark Metrics Spec

## Core metrics

- `throughput`: generated tokens per second
- `TTFT`: time from request admission to the first generated token
- `TPOT`: average time per output token after the first token
- `peak memory`: maximum GPU memory observed during the run

## Reporting rules

- State workload and environment explicitly
- Distinguish measured data from estimates
- Keep model, prompt length, output length, sampling mode, and concurrency aligned when comparing implementations
