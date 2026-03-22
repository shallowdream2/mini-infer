# 博客草稿目录

这里存放待发布的技术博客草稿。

当前高价值草稿入口：

- `09-全项目回顾-从串行Decode到HTTP服务的LLM推理系统实现之路.md`
  - 截止 Phase 8 的阶段性总回顾；已在文首注明后续 Phase 9-12 另有独立文章
- `10-Phase10-Prefix-Caching.md`
  - Prefix Caching 的实现、命中收益和当前仓库复验结果
- `11-Phase11-Speculative-Decoding.md`
  - rejection sampling、KV 一致性问题与 v1 双 forward 的性能边界
- `12-Phase12-CUDA-Graph.md`
  - graph pool、static buffer、与 paged attention/chunked prefill/prefix cache 的兼容性

建议主题方向：

- 为什么要做基于 PagedAttention 的推理引擎
- 如何从 HuggingFace baseline 走到最小推理系统
- Paged KV Cache 的设计与实现
- Continuous Batching 的调度权衡
- 双卡扩展的收益与代价

高质量博客应具备：

- 真实问题驱动
- 真实代码与实验支撑
- 明确设计取舍
- 有坑点和反思
- 语言成熟，不像模板化总结
