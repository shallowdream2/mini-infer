---
name: mini-infer-review
description: Review mini-infer changes with a findings-first code review mindset. Use when the user asks for review, risk analysis, regression checks, benchmark-methodology review, or missing-test detection without making edits.
---

# Mini Infer Review

Only use this skill inside the `mini-infer` repository.

This compatibility alias must follow the same canonical review workflow as Claude `infer-review`.

## Read first

1. `CLAUDE.md`
2. `.claude/skills/infer-review/SKILL.md`
3. `.claude/skills/infer-review/CHECKLIST.md`

## Review mode

- Use the exact output structure and blocking/non-blocking semantics defined by Claude `infer-review`
- Do not edit files while using this skill

## Required output format

Use the Claude `infer-review` output structure, gate, and completion behavior.

## What counts as blocking

- If Claude and Codex docs differ, use the Claude files above

## Finish

- If no blocking issues exist, still preserve the Claude completion semantics
