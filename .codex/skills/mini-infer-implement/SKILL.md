---
name: mini-infer-implement
description: Implement a planned change inside the mini-infer repository. Use when the user wants code changes that follow an existing plan, require minimal relevant validation, and should stay within the current phase scope.
---

# Mini Infer Implement

Only use this skill inside the `mini-infer` repository.

This compatibility alias must follow the same canonical implementation workflow as Claude `infer-implement`.

## Read first

1. `CLAUDE.md`
2. `.claude/rules/workflow.md`
3. `.claude/skills/infer-implement/SKILL.md`

## Preconditions

- Do not relax the Claude requirements around plan-first, prerequisite validation, and repeated-failure fallback

## Implementation rules

- Match the Claude implementation and validation contract exactly

## Validation rules

- Keep the Claude `dry_run` and GPU validation expectations

## What to report after implementation

1. What changed
2. Why the change matches the plan
3. Validation commands and short output summary
4. Remaining risks, gaps, or the next review focus

## Hard stops

- If Claude and Codex docs differ, use the Claude files above
