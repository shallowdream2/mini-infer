---
name: mini-infer-benchmark
description: Design, run, or summarize benchmarks for mini-infer. Use when the user wants workload definition, profiling, result analysis, HF baseline comparison, or a benchmark report with explicit metric and environment rules.
---

# Mini Infer Benchmark

Only use this skill inside the `mini-infer` repository.

This compatibility alias must follow the same canonical benchmark workflow as Claude `infer-benchmark`.

## Read first

1. `CLAUDE.md`
2. `.claude/skills/infer-benchmark/SKILL.md`
3. `.claude/skills/infer-benchmark/METRICS_SPEC.md`

## Benchmark gate

- Keep the Claude hard gate and abnormal-result handling rules

## Every benchmark report must include

- Keep the Claude metric, profiling, and reporting requirements intact

## Default metrics

- If Claude and Codex docs differ, use the Claude files above

## Additional rules for optimization work

- Keep the Claude optimization-stage requirements intact

## Additional rules for attention / kernel replacement

- Keep the Claude attention-replacement requirements intact

## Finish

- Save and update status exactly as Claude `infer-benchmark` requires
