---
name: mini-infer-repo
description: General project-context skill for the mini-infer repository. Use when the user asks to analyze, implement, review, benchmark, summarize, or otherwise work inside mini-infer and you need to load the repository context and route to the right mini-infer workflow.
---

# Mini Infer Repository

Use this skill as the default entry point for work inside the `mini-infer` repository.

## Always read first

1. `CODEX.md`
2. `README.md`
3. `本地资料/环境配置/服务器开发与运行流程.md`

If the task is phase-specific, also read:

- `本地资料/Claude计划/00-长期路线图.md`
- relevant files in `本地资料/实验记录/`
- relevant files in `本地资料/里程碑总结/`

## Routing

After reading the project context, route the task to the right workflow:

- planning -> use `$infer-plan`
- implementation -> use `$infer-implement`
- review -> use `$infer-review`
- benchmark or profiling -> use `$infer-benchmark`
- milestone or phase recap -> use `$infer-summarize`
- technical blog writing -> use `$infer-blog`
- stage closeout or doc gap audit -> use `$infer-archive`

## Core repository rules

- Follow `CLAUDE.md` and `.claude/skills/` as the canonical workflow source
- Prefer `conda run -n ai-infra ...` in agent shells if `conda activate ai-infra` is unstable
- Prefer local absolute model paths plus `HF_HUB_OFFLINE=1`
- Do not claim performance wins without real data
- If weights are missing, say so explicitly
- Keep scope tight and update `README.md` when public project status changes
