---
name: mini-infer-plan
description: Plan a mini-infer task or phase inside the mini-infer repository. Use when the user wants scope definition, file-level changes, prerequisites, acceptance criteria, rollout order, or risk analysis before implementation.
---

# Mini Infer Plan

Only use this skill inside the `mini-infer` repository.

This compatibility alias must follow the same canonical planning workflow as Claude `infer-plan`.

## Read first

1. `CLAUDE.md`
2. `.claude/rules/workflow.md`
3. `.claude/skills/infer-plan/SKILL.md`
4. `.claude/skills/infer-plan/CHECKLIST.md`

## What to produce

Follow the Claude `infer-plan` output structure exactly.

## Planning rules

- If Claude and Codex docs differ, use the Claude files above
- Keep the same gate, checklist, and completion behavior as Claude `infer-plan`

## Output quality bar

- Keep the Claude quality bar intact

## Finish

Append the same itemized self-check required by Claude `infer-plan`.
