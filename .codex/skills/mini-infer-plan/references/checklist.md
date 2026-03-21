# Mini Infer Plan Checklist

Use this checklist at the end of every `mini-infer-plan` output.

## Must cover

- [ ] Motivation references a concrete prior finding, benchmark result, or technical debt
- [ ] Prerequisites include verification commands
- [ ] Scope includes both "do" and "not do"
- [ ] File-level impact is explicit
- [ ] Implementation order has at least 3 independently verifiable steps
- [ ] Each step has a minimal validation method
- [ ] Acceptance criteria include at least 2 quantifiable checks
- [ ] Acceptance criteria include at least 1 correctness check
- [ ] Validation boundary distinguishes dry-run from real GPU / model validation
- [ ] Risks include rollback paths

## Disallowed shortcuts

- [ ] No "functionality normal" style acceptance criteria
- [ ] No "reference previous phase" without naming files and interfaces
- [ ] No oversized plan with too little execution detail
