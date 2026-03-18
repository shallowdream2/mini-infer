---
title: AI Infra学习之旅-服务器环境配置
date: 2026-01-02 14:12:47
tags:
  - Ubuntu Server
  - RTX 4090
  - SSH
  - AI Infrastructure
  - VSCode Remote
  - vLLM
categories: AI Infra
cover: /img/AI-Infra.png
---

## 对当前项目的使用说明

这篇文章记录的是已经执行完成的服务器环境配置过程。对 `mini-infer` 当前开发，默认**直接复用本文已经配置好的 `ai-infra` 环境**，而不是重新创建一个新环境。

当前项目建议直接这样开始：

```bash
conda activate ai-infra
cd ~/mini-infer
python -m pip install -e ".[dev]"
python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py
```

如果需要做真实 benchmark，则继续在同一个环境中执行：

```bash
conda activate ai-infra
nvidia-smi
python benchmarks/benchmark_hf.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 1 --max-new-tokens 32
python benchmarks/benchmark_mini.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 1 --max-new-tokens 32
```

只有在当前环境损坏、缺少关键包，或者你明确要做隔离试验时，才回到本文对应步骤重新安装或修复。

---
## 前言

在上一篇博客中，我在 Kaggle 上成功运行了第一个 vLLM 程序。但 Kaggle 毕竟有时长限制（每周 30 小时），而且每次都要重新配置环境。这次，我有机会使用一台**配备双 RTX 4090 的 Ubuntu 服务器**，这是一次从云端到本地服务器的重要升级！

这篇文章将详细记录我从零开始配置这台服务器的完整过程，包括：

- SSH 免密登录配置
- VSCode Remote SSH 开发环境搭建
- 远程桌面访问（GNOME Remote Desktop）
- Python + vLLM 环境安装
- 实际踩过的坑与解决方案

希望这篇文章能帮助到同样需要配置 AI 开发服务器的朋友们。

---

## 服务器配置信息

### 硬件配置

```
GPU型号: 2 × NVIDIA GeForce RTX 4090
显存: 24GB × 2 = 48GB 总显存
CUDA版本: 12.2
驱动版本: 535.274.02
```

### 系统信息

```
操作系统: Ubuntu 24.04 LTS
用户名: Smarter
主机名: 330B
内网IP: 服务器ip 
```

### 与 Kaggle 对比

| 维度           | 我的服务器 (RTX 4090×2) | Kaggle/Colab       |
| -------------- | ----------------------- | ------------------ |
| **GPU型号**    | RTX 4090 (旗舰)         | ⚠️ P100/T4 (中低端) |
| **显存**       | 48GB                    | ⚠️ 16GB             |
| **GPU时长**    | 无限制                  | ⚠️ 30h/周           |
| **环境持久化** | 永久保存                | ❌ 每次重装         |
| **性能**       | 独享双卡                | ⚠️ 共享单卡         |
| **适合长实验** |                         | ❌                  |

**结论**: 服务器配置远超 Kaggle，应该优先使用服务器进行学习和开发。

---

## 第一阶段：SSH 连接配置

### Step 1: 首次 SSH 连接测试

在 Windows PowerShell 中执行：

```powershell
# 首次连接（替换为你的用户名@服务器IP）
ssh Smarter@服务器ip

# 首次连接会提示：
# The authenticity of host '192.168.50.58' can't be established.
# Are you sure you want to continue connecting (yes/no)?
# 输入：yes

# 然后输入密码
```

**成功标志**：看到以下提示符说明连接成功

```bash
Smarter@330B:~$
```

### Step 2: 验证 GPU 状态

连接成功后，立即验证 GPU：

```bash
nvidia-smi
```

**我的实际输出**：

![image-20260102141851912](https://cdn.jsdelivr.net/gh/psmarter/blog-imgs/AI-Infra%E5%AD%A6%E4%B9%A0%E4%B9%8B%E6%97%85-%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/2026-01-02-14-39-22-d0d150.png)

看到两张 RTX 4090，CUDA 12.2，驱动正常，说明环境 OK！

### Step 3: 配置 SSH 密钥免密登录

每次输入密码很麻烦，配置公钥登录可以一劳永逸。

#### 在 Windows 本地生成 SSH 密钥

打开 PowerShell：

```powershell
# 生成 SSH 密钥对（如果还没有）
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 提示：Enter file in which to save the key
# 直接回车（使用默认路径 C:\Users\pc\.ssh\id_rsa）

# 提示：Enter passphrase
# 直接回车（不设置密码，方便使用）
```

#### 将公钥复制到服务器

```powershell
# 读取公钥内容
$pub = Get-Content C:\Users\pc\.ssh\id_rsa.pub

# 将公钥追加到服务器的 authorized_keys
ssh Smarter@服务器ip "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pub' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

> 💡 这一步会要求输入最后一次服务器密码，之后就可以免密登录了。

#### 测试免密登录

```powershell
ssh Smarter@服务器ip
```

如果直接登录不再要求密码，说明配置成功！

---

## 第二阶段：VSCode Remote SSH 配置

VSCode Remote SSH 是最重要的开发工具，可以让我在 Windows 上直接编辑服务器上的代码。

### Step 1: 安装 VSCode 插件

在 VSCode 中安装以下插件：

1. **Remote - SSH**  (必装)
2. **Remote - SSH: Editing Configuration Files**
3. **Python**
4. **Jupyter**

### Step 2: 配置 SSH Config 文件

1. 按 `F1`，输入 `Remote-SSH: Open SSH Configuration File`
2. 选择 `C:\Users\pc\.ssh\config`
3. 添加以下配置：

```ssh-config
# Ubuntu AI Infra服务器 (RTX 4090×2)
Host ai-server
    HostName 服务器ip
    User Smarter
    Port 22
    IdentityFile C:\Users\pc\.ssh\id_rsa
    IdentitiesOnly yes
    PreferredAuthentications publickey
    ForwardAgent yes
```

**配置说明**：

- `IdentitiesOnly yes`: 只使用指定的密钥
- `PreferredAuthentications publickey`: 优先使用公钥认证
- `ForwardAgent yes`: 允许 SSH Agent 转发

### Step 3: 连接到服务器

1. 按 `F1`，输入 `Remote-SSH: Connect to Host`
2. 选择 `ai-server`
3. 等待连接（首次会安装 VSCode Server，约 1 分钟）
4. **成功标志**：左下角显示 `SSH: ai-server` 

### Step 4: 在 VSCode 中打开服务器文件夹

连接成功后：

1. 点击 `File` → `Open Folder`
2. 选择 `/home/Smarter` 或你的项目目录
3. 现在可以直接在 VSCode 中编辑服务器上的文件了！

---

## 第三阶段：远程桌面配置

有时候需要图形界面，我配置了 GNOME Remote Desktop。

### 为什么选择 GNOME Remote Desktop？

-  Ubuntu 24.04 自带，无需额外安装
-  支持无显示器远程登录
-  Windows 自带 mstsc 即可连接
-  比 XRDP 更稳定

### Step 1: 安装和启用 GNOME Remote Desktop

在服务器上执行（需要管理员权限）：

```bash
# 1. 安装组件（Ubuntu 24.04 通常已安装）
sudo apt update
sudo apt install -y gnome-remote-desktop

# 2. 生成 TLS 证书（必需）
sudo -u gnome-remote-desktop mkdir -p ~gnome-remote-desktop/.local/share/gnome-remote-desktop

sudo openssl req -x509 -newkey rsa:4096 -nodes -days 3650 -sha256 \
  -keyout ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.key \
  -out ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.crt \
  -subj "/CN=服务器ip"

# 3. 设置文件权限
sudo chown gnome-remote-desktop:gnome-remote-desktop \
  ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.key \
  ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.crt

sudo chmod 600 ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.key
sudo chmod 644 ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.crt
```

### Step 2: 配置 RDP 认证

```bash
# 禁用 RDP（如果之前启用过）
sudo grdctl --system rdp disable 2>/dev/null || true

# 设置 TLS 证书
sudo grdctl --system rdp set-tls-key ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.key
sudo grdctl --system rdp set-tls-cert ~gnome-remote-desktop/.local/share/gnome-remote-desktop/tls.crt

# 设置 RDP 入口账号密码（这不是 Linux 用户密码）
sudo grdctl --system rdp set-credentials rdpuser '你的强密码'

# 启用 RDP
sudo grdctl --system rdp enable
```

### Step 3: 启动服务并验证

```bash
# 启动并设置开机自启
sudo systemctl enable --now gnome-remote-desktop.service

# 重启服务
sudo systemctl restart gnome-remote-desktop.service

# 验证 3389 端口是否监听
sudo ss -lnptu | grep 3389

# 查看状态
sudo grdctl --system status
```

### Step 4: Windows 端连接

1. 按 `Win + R`，输入 `mstsc`
2. 计算机填写：`服务器ip`
3. 首次认证输入：
   - 用户名：`rdpuser`
   - 密码：你设置的密码
4. 进入 GNOME 登录界面后，使用 Linux 用户 `Smarter` 的系统密码登录

> ⚠️ **重要**：确保 `Smarter` 用户设置了系统密码：
>
> ```bash
> sudo passwd Smarter
> ```

### Step 5: 保存 RDP 凭据（避免每次输入）

在 Windows 中：

1. 搜索并打开 **凭据管理器**
2. 进入 **Windows 凭据**
3. 点击 **添加 Windows 凭据**
4. 地址填：`TERMSRV/服务器ip`
5. 用户名填：`rdpuser`
6. 密码填你的密码 → 保存

---

## 第四阶段：Python 环境配置

### Step 1: 检查 Conda 是否已安装

在 VSCode 终端（或 SSH 连接）中：

```bash
conda --version
```

- 如果显示版本号 → 已安装 
- 如果报错 `command not found` → 需要安装

### Step 2A: 如果已有 Conda

```bash
# 1. 检查 CUDA 版本
nvidia-smi | grep "CUDA Version"
# 输出: CUDA Version: 12.2

# 2. 检查现有环境
conda env list

# 3. 创建 AI Infra 环境（如果不存在）
conda create -n ai-infra python=3.10 -y

# 4. 激活环境
conda activate ai-infra

# 5. 安装 PyTorch（CUDA 12.2 使用 cu121）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 6. 验证 GPU 可用性
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

**期望输出**：

```
PyTorch: 2.1.2+cu121
CUDA可用: True
GPU数量: 2
```

### Step 2B: 如果需要安装 Conda

```bash
# 1. 下载 Miniconda（比 Anaconda 更轻量）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 2. 安装
bash Miniconda3-latest-Linux-x86_64.sh
# 提示: Do you accept the license terms? → yes
# 提示: installation location → 回车（使用默认）
# 提示: Do you wish to update your shell profile → yes

# 3. 重新加载 shell
source ~/.bashrc

# 4. 验证安装
conda --version

# 5. 然后回到 Step 2A 的步骤 3 继续
```

### Step 3: 安装基础科学计算包

```bash
# 确保在 ai-infra 环境中
conda activate ai-infra

# 安装基础包（指定版本避免冲突）
pip install numpy==1.24.3 pandas matplotlib jupyterlab
pip install transformers==4.36.2 accelerate==0.25.0

# 验证安装
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 第五阶段：vLLM 安装与测试

### 版本兼容性说明

> ⚠️ **重要**：不要随意升级版本，容易导致 CUDA 不兼容！

```
推荐版本组合（2026年1月验证）：
- PyTorch 2.1.2 (cu121)
- vLLM 0.2.7 (稳定版)
- Triton 2.1.0 (vLLM依赖)
- Flash-Attention 2.3.6 (可选，编译困难，不推荐)
```

### 保守安装方案（推荐）

```bash
# 1. 激活环境
conda activate ai-infra

# 2. 安装 vLLM（会自动安装 Triton 等依赖）
pip install vllm==0.2.7

# 3. 验证 vLLM
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
# 输出: vLLM版本: 0.2.7

# 4. 验证 Triton（vLLM 会自动安装）
python -c "import triton; print(f'Triton版本: {triton.__version__}')"

# 5. 测试 vLLM 是否可用
python -c "from vllm import LLM; print('vLLM导入成功 ')"
```

### 最终环境验证

运行完整验证脚本：

```bash
python <<EOF
import torch
import vllm
import triton

print("=" * 50)
print("环境验证")
print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
print(f"\nvLLM版本: {vllm.__version__}")
print(f"Triton版本: {triton.__version__}")

print("\n 所有核心组件安装成功！")
EOF
```

**期望输出**：

```
==================================================
环境验证
==================================================
PyTorch版本: 2.1.2+cu121
CUDA可用: True
CUDA版本: 12.1
GPU数量: 2
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

vLLM版本: 0.2.7
Triton版本: 2.1.0

 所有核心组件安装成功！
```

---

## 第六阶段：运行第一个 vLLM 程序

### 创建测试文件

在 VSCode 中创建 `~/test_vllm.py`：

```python
from vllm import LLM, SamplingParams
import torch
import time

# 显示 GPU 信息
print("=" * 80)
print("GPU 配置信息")
print("=" * 80)
print(f"可用GPU数量: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
print(f"GPU 1: {torch.cuda.get_device_name(1)}")
print("=" * 80)

# 加载小模型测试（只有 125M 参数，几秒加载）
print("\n🔄 正在加载模型...")
llm = LLM(
    model="facebook/opt-125m",
    trust_remote_code=True,
    gpu_memory_utilization=0.5  # 只用 50% 显存，不影响其他人
)
print(" 模型加载完成！\n")

# 准备输入
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "AI Infrastructure is"
]

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50
)

# 执行推理
print("🚀 开始推理...\n")
outputs = llm.generate(prompts, sampling_params)

# 查看结果
print("=" * 80)
print("📊 推理结果")
print("=" * 80)
for output in outputs:
    print(f"\n提示词: {output.prompt}")
    print(f"生成结果: {output.outputs[0].text}")
    print("-" * 80)

print("\n 成功运行第一个 vLLM 程序！")

# 性能测试
print("\n⏱️  性能测试开始...")
test_prompts = ["Explain AI in simple terms"] * 10

start = time.time()
outputs = llm.generate(test_prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed

print("\n" + "=" * 80)
print("📈 服务器性能测试结果")
print("=" * 80)
print(f"模型:        OPT-125M")
print(f"GPU:         RTX 4090 (单卡)")
print(f"总耗时:      {elapsed:.2f} 秒")
print(f"总 Token 数: {total_tokens}")
print(f"吞吐量:      {throughput:.2f} tokens/秒")
print("=" * 80)
```

### 运行测试

```bash
# 指定使用 GPU 0（避免占用他人 GPU）
CUDA_VISIBLE_DEVICES=0 python ~/test_vllm.py
```

### 我的运行结果

![image-20260102143458380](https://cdn.jsdelivr.net/gh/psmarter/blog-imgs/AI-Infra%E5%AD%A6%E4%B9%A0%E4%B9%8B%E6%97%85-%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/2026-01-02-14-39-22-7ce4ec.png)

> 💡 **性能对比**：
>
> - Kaggle P100: ~1539 tokens/秒
> - 我的 RTX 4090: ~5818 tokens/秒
> - **提升约 278%！** 🚀

---

## 踩过的坑与解决方案

### 问题 1: VSCode Remote SSH 每次要输密码

**现象**: VSCode 连接服务器时总是弹窗要求输入密码

**解决方案**: 配置 SSH 公钥免密登录（见"第一阶段 Step 3"）

关键步骤：

```powershell
# Windows 本地
$pub = Get-Content C:\Users\pc\.ssh\id_rsa.pub
ssh Smarter@服务器ip "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pub' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

并在 SSH config 中添加：

```
IdentitiesOnly yes
PreferredAuthentications publickey
```

### 问题 2: GNOME Remote Desktop 无法连接

**现象**: Windows 端 mstsc 连接 3389 端口无响应

**原因**:

1. 没有生成 TLS 证书
2. 没有设置 RDP 认证凭据
3. XRDP 占用了 3389 端口

**解决方案**:

```bash
# 1. 停止并卸载 XRDP（如果有）
sudo systemctl disable --now xrdp xrdp-sesman
sudo apt purge -y xrdp xorgxrdp

# 2. 生成 TLS 证书（见"第三阶段 Step 1"）

# 3. 设置 RDP 凭据
sudo grdctl --system rdp set-credentials rdpuser '强密码'
sudo grdctl --system rdp enable

# 4. 重启服务
sudo systemctl restart gnome-remote-desktop.service

# 5. 验证端口监听
sudo ss -lnptu | grep 3389
```

### 问题 3: 远程桌面提示"已有会话运行"

**现象**: 尝试远程登录时提示"该用户已有会话"

**原因**: 之前的远程会话没有正常退出，系统认为仍在运行

**解决方案**:

```bash
# 1. 查看当前会话
loginctl list-sessions
loginctl user-status Smarter

# 2. 终止该用户所有会话
sudo loginctl terminate-user Smarter

# 3. 重启远程桌面服务
sudo systemctl restart gnome-remote-desktop.service
sudo systemctl restart gdm3

# 4. 再尝试连接
```

### 问题 4: vLLM 导入报错 CUDA 版本不匹配

**现象**: `import vllm` 报错 "CUDA version mismatch"

**原因**: PyTorch 的 CUDA 版本与系统 CUDA 不匹配

**解决方案**:

```bash
# 1. 卸载 PyTorch
pip uninstall torch torchvision -y

# 2. 重新安装匹配的版本（CUDA 12.2 使用 cu121）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3. 验证
python -c "import torch; print(torch.version.cuda)"
# 应该输出: 12.1 (PyTorch 对 CUDA 12.x 通用)
```

### 问题 5: Conda 命令找不到

**现象**: `conda: command not found`

**解决方案**:

```bash
# 重新加载 bashrc
source ~/.bashrc

# 如果还是不行，手动添加到 PATH
export PATH="$HOME/anaconda3/bin:$PATH"

# 或者添加到 ~/.bashrc
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 问题 6: Clash 代理端口被占用

**现象**: 想重启 Clash 但提示端口 7890 被占用

**解决方案**:

```bash
# 1. 查看占用端口的进程
sudo lsof -iTCP:7890 -sTCP:LISTEN

# 输出示例:
# COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# clash   3528110  Smarter    9u  IPv4 xxx      0t0  TCP localhost:7890 (LISTEN)

# 2. 终止该进程
sudo kill 3528110

# 3. 重新启动 Clash
```

---

## 开发工作流程

配置完成后，我的日常开发流程是这样的：

### 方案：VSCode Remote SSH

```
┌─────────────────┐         SSH        ┌─────────────────┐
│  Windows笔记本   │ ◄────────────────► │ Ubuntu服务器    │
│                 │                    │                 │
│  VSCode界面     │                    │  实际执行代码    │
│  编辑器         │                    │  双GPU计算      │
│  浏览器         │                    │  永久存储       │
└─────────────────┘                    └─────────────────┘
       本地                                   云端
```

**日常步骤**:

1. **打开 VSCode 连接服务器**

   ```
   VSCode → F1 → Remote-SSH: Connect to Host → ai-server
   ```

2. **编辑代码**（和本地一样）

   - 在服务器上直接编辑项目文件
   - VSCode 体验和本地完全一样
   - 代码保存在服务器上，永久保存 

3. **运行实验**（服务器 GPU）

   ```bash
   # VSCode 集成终端
   conda activate ai-infra
   
   # 检查 GPU 状态
   nvidia-smi
   
   # 指定使用 GPU 0
   CUDA_VISIBLE_DEVICES=0 python train.py
   
   # GPU 计算，无时长限制 
   ```

4. **查看结果**

   - 结果保存在服务器
   - 可以通过 VSCode 直接查看图片/日志
   - 或者下载到本地

---

## GPU 资源管理

### 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者安装 nvitop（更友好的界面）
pip install nvitop
nvitop
```

### 指定 GPU 运行代码

```bash
# 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py

# 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py

# 同时使用两张 GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### Python 代码中指定

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只用第一张 GPU

# 或者
import torch
device = torch.device('cuda:0')  # GPU 0
# device = torch.device('cuda:1')  # GPU 1
```

### 多人共享礼仪

如果服务器是多人共享的：

```bash
# 每次使用前检查 GPU 状态
nvidia-smi

# 看 Processes 列：
# - GPU 0 空闲 → CUDA_VISIBLE_DEVICES=0
# - GPU 1 空闲 → CUDA_VISIBLE_DEVICES=1
# - 都在用 → 等待或协调
```

---

## 长时间任务：使用 screen

服务器的优势之一就是可以运行长时间任务，即使关闭本地电脑也不影响。

### 安装 screen

```bash
sudo apt install screen
```

### 使用方法

```bash
# 1. 创建 session
screen -S my_experiment

# 2. 运行长时间任务
conda activate ai-infra
python long_training.py

# 3. 断开（任务继续运行）
# 按 Ctrl+A，然后按 D

# 4. 重新连接
screen -r my_experiment

# 5. 列出所有 session
screen -ls

# 6. 终止 session（在 session 内部）
exit
```

**优势**：关闭 Windows 电脑，SSH 断开，任务照样运行！

---

## 总结与收获

### 完成的配置

- [x]  SSH 免密登录配置
- [x]  VSCode Remote SSH 环境
- [x]  GNOME Remote Desktop 远程桌面
- [x]  Conda 环境创建 (ai-infra)
- [x]  PyTorch 2.1.2 + CUDA 12.1 安装
- [x]  vLLM 0.2.7 安装与测试
- [x]  第一个 vLLM 程序运行成功
- [x]  性能测试：2276 tokens/秒

### 性能对比

| 平台         | GPU      | 吞吐量 (tokens/s) | 成本               |
| ------------ | -------- | ----------------- | ------------------ |
| Kaggle       | P100     | ~1539             | 免费（30h/周）     |
| 我的服务器   | RTX 4090 | ~5818             | 内网服务器（无限） |
| **性能提升** | -        | **+278%**         | -                  |

---

## 写在最后

如果你也有机会使用服务器进行 AI 学习，我的建议是：

1. **优先配置好 SSH 免密登录** - 这是一切的基础
2. **VSCode Remote SSH 是最佳开发环境** - 比本地+rsync 方便太多
3. **认真做版本管理** - 不要随意升级，稳定的版本组合很重要
4. **做好 GPU 资源管理** - 尤其是共享服务器，要有礼貌
5. **使用 screen 管理长任务** - 充分利用服务器优势

---

## 参考资料

- [Ubuntu GNOME Remote Desktop 官方文档](https://help.ubuntu.com/stable/ubuntu-help/sharing-desktop.html)
- [VSCode Remote SSH 文档](https://code.visualstudio.com/docs/remote/ssh)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)

---

**如果这篇文章对你有帮助，欢迎点赞和分享！有任何问题也欢迎在评论区交流。** 🎉

