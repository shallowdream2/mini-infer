# 从 HuggingFace generate() 到自己的推理引擎：Phase 1 最小链路复盘

这篇文章是 mini-infer 项目的第一篇技术复盘，记录如何在不改模型结构的前提下，用最少的代码搭起一条可以跑 Qwen2.5-7B 的推理主链路，以及这条链路和 HuggingFace baseline 的真实性能差距是多少、差在哪里。

---

## 一、为什么不直接用 HuggingFace generate()

问题不在于能不能用，而在于用了之后你什么都不知道。

`model.generate()` 是一个黑盒：它帮你处理 KV cache、attention mask、采样、EOS 检测，还支持 beam search 和各种解码策略。但正因如此，你无法控制调度粒度，无法在多请求之间动态分配显存，也无法在一个 batch 的 decode 步骤中间插入新请求。

推理系统的核心挑战不是跑通一条请求，而是**同时处理多条不同长度的请求，并在显存、吞吐和延迟之间做权衡**。这需要你自己管理 KV cache、自己调度，而不是依赖 `generate()` 的固定流程。

mini-infer 的 Phase 1 目标是：用最小代码搭出一条可验证的推理主链路，并和 HF baseline 做定量对比，明确当前差距和原因。

---

## 二、设计：用 HF 的 past_key_values 作为 Phase 1 的 KV cache

Phase 1 没有自己管理 KV cache 的 GPU tensor，而是直接借用 HuggingFace 模型的 `past_key_values` 机制：

```
prefill:  model(input_ids=prompt_tokens, use_cache=True)
          → logits + past_key_values（存在 dict 里）

decode:   model(input_ids=[[last_token]], past_key_values=..., use_cache=True)
          → logits + updated past_key_values
```

整体架构分四层：

```
LLMEngine.generate()
  └── Scheduler.get_next_batch()        # FIFO，返回一批 RequestState
  └── ModelRunner.prefill(batch)        # 逐条 prefill，存 past_key_values
  └── while unfinished:
        ModelRunner.decode_step(batch)  # 逐条 decode，更新 past_key_values
  └── tokenizer.decode(generated_ids)  # 整体 decode，避免子词拼接问题
```

`RequestState` 保存每个请求的 prompt token ids、已生成 token ids 和完成状态。`past_key_values` 按 `request_id` 存在 `ModelRunner._past_kv` 这个 dict 里，请求结束时 `free_request()` 删掉引用让 Python GC 回收显存。

这个设计的优点是代码极简，缺点已经写进代码注释：

```python
# Phase 1：用 HuggingFace past_key_values 存储每个请求的 KV，Phase 2 替换为分页 GPU tensor
self._past_kv: dict[str, tuple] = {}
```

---

## 三、实现中踩到的坑

### 坑 1：transformers 版本不支持 Qwen2

初始环境的 transformers 是 4.36.2，Qwen2Tokenizer 从 4.40.0 才开始支持。直接报错：

```
ValueError: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.
```

升级到 5.x 也不行——transformers 5.x 引入了 `torch.utils._pytree.register_pytree_node`，这个 API 在 PyTorch 2.4 才有，而项目环境是 PyTorch 2.1.2：

```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

最终锁定在 4.43.4：支持 Qwen2，兼容 PyTorch 2.1.2。

这个版本约束需要在 `pyproject.toml` 里明确，否则 `pip install -e .` 会拉到 5.x 然后启动时崩溃。

### 坑 2：huggingface_hub 的 sha256 校验在大文件上会挂死

用 `snapshot_download` 下载 Qwen2.5-7B（约 15GB）时，`model-00002-of-00004.safetensors`（3.6GB）卡了超过 5 小时：进程存活，有网络连接，但文件不再增长。

原因：`snapshot_download` 在 `.incomplete` 文件下载完成后，会做 sha256 校验再重命名。这个校验在后台进程中某些条件下会挂起。

判断文件是否完整的方法是比较文件大小和服务器的 `Content-Length`：

```bash
curl -sI -L "https://huggingface.co/.../model-00002-of-00004.safetensors" | grep content-length
# Content-Length: 3864726352

stat --printf="%s\n" ~/.../model-00002-of-00004.safetensors.incomplete
# 3864726352
```

两个数字一致，说明内容已完整下载，直接 `cp` 重命名即可，跳过卡死的校验步骤。

### 坑 3：pyproject.toml 的包发现问题

项目根目录下有 `本地资料/` 这个中文目录，setuptools 的 flat-layout 包发现逻辑会把它识别为一个 Python 包，导致 `pip install -e .` 报错：

```
error: Multiple top-level packages discovered in a flat-layout: ['本地资料', 'mini_infer']
```

修复方式是显式指定只包含 `mini_infer`：

```toml
[tool.setuptools.packages.find]
include = ["mini_infer*"]
```

### 坑 4：eos_token_id 的 falsy 判断

代码里有一行：

```python
self.eos_token_id = self.tokenizer.eos_token_id or 0
```

Qwen2.5 的 eos_token_id 是 151645，没有问题。但如果某个模型的 eos_token_id 是 0，`0 or 0` 在 Python 里因为 falsy 仍然返回 0，勉强没有错；但如果 tokenizer 返回 `None`，`None or 0` 会静默地把 EOS 设为 0，导致 token_id=0 就停止生成。正确写法：

```python
self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
```

这个问题在 code review 时发现，Qwen2.5 下不会触发，但换模型时是隐患。

---

## 四、实验：和 HuggingFace baseline 的真实差距

环境：Ubuntu 24.04，单卡 RTX 4090，CUDA 12.2，PyTorch 2.1.2，transformers 4.43.4，Qwen2.5-7B-Instruct，float16，greedy decode，max_new_tokens=128。

**HF Baseline（transformers.generate，整批并行）：**

| batch_size | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Peak Mem (GB) |
|-----------|-----------|--------------|-------------------|---------------|
| 1 | 19.0 | 17.78 | 56.2 | 15.78 |
| 4 | 20.8 | 18.99 | 210.4 | 15.81 |
| 8 | 23.6 | 19.53 | 408.9 | 15.88 |

**mini-infer Phase 1（串行 past_key_values）：**

| batch_size | Throughput (tok/s) | Peak Mem (GB) |
|-----------|-------------------|---------------|
| 1 | 56.4 | 15.78 |
| 4 | 56.4 | 15.81 |
| 8 | 56.3 | 15.84 |

**差距倍数：**

| batch_size | HF | mini-infer Ph1 | 差距 |
|-----------|-----|---------------|------|
| 1 | 56.2 | 56.4 | **持平** |
| 4 | 210.4 | 56.4 | **3.7×** |
| 8 | 408.9 | 56.3 | **7.3×** |

### 数据说明了什么

batch=1 时两者几乎相同——这说明单请求下的实现是正确的，forward 路径等价。

batch=4 时 mini-infer 的吞吐没有增长，还是 56 tok/s。原因直接写在 `decode_step` 里：

```python
for state in states:          # 串行循环
    if state.finished:
        continue
    out = self.model(
        input_ids=[[last_token]],      # 每次只送 1 个 token
        past_key_values=past_kv,       # 每个请求独立的 KV
        use_cache=True
    )
```

4 条请求 = 4 次独立的 GPU forward，GPU 利用率极低。HF baseline 的 `generate()` 把 4 条请求合并成一个 `[4, 1]` 的 input_ids 做一次 forward，计算效率随 batch 线性提升。

显存方面两者几乎相同（差异 < 0.1 GB）：Phase 1 借用了 HF 的 `past_key_values` 存储结构，内存布局和 baseline 一致。

### 数据的局限性

- prompt 很短（10-15 tokens），不代表真实生产场景
- mini-infer Phase 1 没有测 TTFT 和 TPOT（benchmark 缺少分阶段计时）
- 单次测量，没有多次取平均
- HF baseline 的 batch>1 使用了 right-padding，对 decoder-only 模型的生成质量有轻微影响

---

## 五、设计取舍的反思

**为什么 Phase 1 不直接做 batch decode？**

因为在没有 Paged KV Cache 的情况下做 batch decode，需要把不同长度的请求的 past_key_values padding 到相同长度，然后合并成一个 batch。这在实现上并不复杂，但如果 KV cache 的数据结构和 Phase 2 不兼容，等于做了两遍。

Phase 1 的设计目标是"链路跑通，接口正确"，不是"性能最优"。这个决定让 Phase 1 的代码极简且可测，差距被数据量化清楚，Phase 2 的改进方向也因此明确。

**为什么不用 `device_map="cuda:0"` 而是 `.to(device)`？**

先 load 到 CPU 再 to(device) 的方式，在加载 7B 模型时会短暂消耗 ~14GB CPU RAM + ~14GB GPU VRAM。当前服务器内存够用所以没有触发问题，但这是已知风险。正确做法是用 `device_map` 让模型直接 mmap 到 GPU。这是 Phase 2 需要修复的点之一。

---

## 六、下一步

Phase 2 要解决的核心问题只有一个：**把串行的 `for state: model(1 token)` 改成真正的 batch forward**。

配合这个改动，需要把 KV cache 从每请求独立的 `past_key_values` dict 改为预分配的 GPU block tensor 池，并引入 BlockTable 来管理不同请求对物理 block 的映射。这就是 Paged KV Cache 的核心思路。

当前数据给了一个明确的改进目标：**batch=4 的吞吐从 56 tok/s 提升到接近 HF baseline 的 210 tok/s**。如果 Phase 2 做完之后这个数字还差很多，说明 batch decode 的实现有问题。

---

*代码：[mini-infer](https://github.com/psmarter/mini-infer)*
*实验数据来源：`本地资料/实验记录/2026-03-19-qwen-baseline-benchmark.md`*
