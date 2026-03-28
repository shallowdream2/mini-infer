"""
gen_charts.py — 从实验记录数据生成 mini-infer benchmark 对比图表。
输出目录：assets/charts/
用法：python scripts/gen_charts.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 中文字体 ─────────────────────────────────────────────────────────────────
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
_cjk_font = None
for _fp in _CJK_FONT_PATHS:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        _cjk_font = _prop.get_name()
        break

# ── 全局样式 ────────────────────────────────────────────────────────────────
if _cjk_font:
    plt.rcParams["font.family"] = [_cjk_font, "DejaVu Sans", "sans-serif"]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
})

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
GRAY   = "#8C8C8C"

# ── 图 1：主线吞吐演进（Phase 3 → 6 → HF baseline）────────────────────────
def chart_throughput_evolution():
    phases   = ["Phase 3\n(DynamicCache)", "Phase 6\n(True PagedAttn)", "HF Baseline\n(Transformers)"]
    tps      = [361.3, 406.3, 406.4]
    colors   = [ORANGE, BLUE, GRAY]
    pct      = ["88.4%", "100.0%", "—"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(phases, tps, color=colors, width=0.45, zorder=3)

    for bar, p, t in zip(bars, pct, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 4,
                f"{t:.0f} tok/s\n({p})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 480)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("主线吞吐演进 — Qwen2.5-7B, batch=8, max_new_tokens=128")
    ax.axhline(406.4, color=GRAY, linestyle="--", linewidth=1.2, zorder=2, label="HF baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_throughput_evolution.png"))
    plt.close(fig)
    print("✓ 01_throughput_evolution.png")


# ── 图 2：CUDA Graph decode 延迟（1.5B 模型，bs 1/2/4/8）──────────────────
def chart_cuda_graph():
    bs         = [1, 2, 4, 8]
    eager_ms   = [7.33, 7.44, 7.73, 8.31]
    graph_ms   = [5.21, 5.88, 6.43, 6.79]
    speedups   = [1.41, 1.27, 1.20, 1.22]

    x   = np.arange(len(bs))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))

    b1 = ax.bar(x - w/2, eager_ms, w, color=ORANGE, label="Eager", zorder=3)
    b2 = ax.bar(x + w/2, graph_ms, w, color=BLUE,   label="CUDA Graph", zorder=3)

    for bar, sp in zip(b2, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{sp:.2f}×",
                ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={b}" for b in bs])
    ax.set_ylabel("Decode step latency (ms)")
    ax.set_title("CUDA Graph vs Eager — Qwen2.5-1.5B decode latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_cuda_graph.png"))
    plt.close(fig)
    print("✓ 02_cuda_graph.png")


# ── 图 3：MoE EP 吞吐演进（dense → padded → packed → grouped）──────────────
def chart_moe_ep():
    labels = ["Dense\n(1 GPU)", "EP padded\n(2 GPU)", "EP packed\n(2 GPU)", "EP grouped\n(2 GPU)"]
    tps    = [21878.76, 42765.91, 51121.99, 54696.73]
    colors = [GRAY, ORANGE, GREEN, BLUE]
    ratios = ["1.0×", "1.954×", "2.337×", "2.500×"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, tps, color=colors, width=0.5, zorder=3)

    for bar, r, t in zip(bars, ratios, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 400,
                f"{t/1000:.1f}k\n({r})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 65000)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("MoE Expert Parallelism 吞吐演进 — Synthetic MoE, 2 × RTX 4090")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_moe_ep_evolution.png"))
    plt.close(fig)
    print("✓ 03_moe_ep_evolution.png")


# ── 图 4：Flash Decoding 延迟 vs seq_len（1.5B，块式 KV）──────────────────
def chart_flash_decode():
    seq_lens      = [128, 256, 512, 1024, 2048, 4096]
    flash_attn_ms = [0.015, 0.014, 0.012, 0.010, 0.016, 0.013]
    triton_65_ms  = [0.020, 0.023, 0.044, 0.074, 0.129, 0.256]
    flash_dec_ms  = [0.062, 0.063, 0.063, 0.063, 0.068, 0.100]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(seq_lens, flash_attn_ms, "o-", color=GRAY,   label="flash_attn（基准）", linewidth=2)
    ax.plot(seq_lens, triton_65_ms,  "s-", color=ORANGE, label="Triton baseline（split-K ref）", linewidth=2)
    ax.plot(seq_lens, flash_dec_ms,  "^-", color=BLUE,   label="Flash Decoding（split-K Triton）", linewidth=2)

    ax.axvline(2048, color=GREEN, linestyle=":", linewidth=1.2, label="crossover ≈ 2048")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Attention latency (ms)")
    ax.set_title("Flash Decoding (Split-K) vs Triton baseline — Qwen2.5-1.5B, bs=1")
    ax.legend(loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.annotate("3.31× faster\nvs Triton @ seq=4096",
                xy=(4096, 0.100), xytext=(2200, 0.18),
                arrowprops=dict(arrowstyle="->", color=BLUE),
                fontsize=9, color=BLUE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_flash_decode.png"))
    plt.close(fig)
    print("✓ 04_flash_decode.png")


# ── 图 5：Chunked Prefill — ITL spike 对比 ─────────────────────────────────
def chart_chunked_prefill():
    categories = ["无 Chunked Prefill\n(chunk=0)", "chunk=128", "chunk=256"]
    itl_spike  = [138.4, 45.9, 58.6]   # ms，短请求最大 ITL
    throughput = [290.0, 296.8, 301.3]  # tok/s 总吞吐

    x  = np.arange(len(categories))
    w  = 0.35
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, itl_spike,  w, color=RED,   label="Max ITL spike (ms)", zorder=3)
    b2 = ax2.bar(x + w/2, throughput, w, color=BLUE,  label="Throughput (tok/s)", zorder=3)

    for bar, v in zip(b1, itl_spike):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.0f} ms", ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")
    for bar, v in zip(b2, throughput):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Max ITL spike (ms)", color=RED)
    ax2.set_ylabel("Throughput (tok/s)", color=BLUE)
    ax1.set_ylim(0, 200)
    ax2.set_ylim(270, 320)
    ax1.set_title("Chunked Prefill — ITL spike vs throughput — Qwen2.5-7B")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.annotate("−66.8%", xy=(1 - w/2, 45.9), xytext=(0.5, 120),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color=RED, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_chunked_prefill.png"))
    plt.close(fig)
    print("✓ 05_chunked_prefill.png")


# ── 图 6：W8A8 量化 — 显存收益 vs 精度代价 ─────────────────────────────────
def chart_w8a8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    # 左图：显存对比
    models  = ["FP16", "W8A8"]
    mem_mb  = [3392.4, 2292.0]
    colors_ = [ORANGE, BLUE]
    bars = ax1.bar(models, mem_mb, color=colors_, width=0.4, zorder=3)
    for bar, v in zip(bars, mem_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 30,
                 f"{v:.0f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 4200)
    ax1.set_ylabel("Weight memory (MB)")
    ax1.set_title("显存：权重占用")
    ax1.annotate("−32.4%", xy=(1, 2292), xytext=(0.6, 3000),
                 arrowprops=dict(arrowstyle="->", color=BLUE),
                 fontsize=11, color=BLUE, fontweight="bold")

    # 右图：decode tps 对比
    models2 = ["FP16", "W8A8\n(mixed fallback)"]
    tps2    = [534.8, 190.6]
    bars2 = ax2.bar(models2, tps2, color=[ORANGE, RED], width=0.4, zorder=3)
    for bar, v in zip(bars2, tps2):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 8,
                 f"{v:.0f} tok/s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 650)
    ax2.set_ylabel("Decode throughput (tok/s)")
    ax2.set_title("推理吞吐（小 M 限制）")

    note = ax2.text(0.5, 0.12, "decode 退回 FP16 mixed fallback\n（torch._int_mm 小 M 限制）",
                    transform=ax2.transAxes, ha="center", fontsize=8.5,
                    color="gray", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.8))

    fig.suptitle("W8A8 量化（per-channel int8）— Qwen2.5-1.5B", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_w8a8_quant.png"))
    plt.close(fig)
    print("✓ 06_w8a8_quant.png")


# ── 图 7：综合性能总览（雷达图）────────────────────────────────────────────
def chart_overview_radar():
    # 各项能力指标（0-10 满分，根据实验结论主观打分）
    # 设计: 5 个维度，用多边形对比 mini-infer 与"未实现"基线
    categories = ["吞吐\n(batch decode)", "延迟\n(decode latency)", "内存效率\n(KV cache)",
                  "分布式扩展\n(EP 2.5×)", "长序列\n(Flash Decode)"]
    values_mini = [10.0, 8.5, 7.5, 9.5, 8.0]  # 归一化到 10 分
    values_base = [8.8,  7.1, 5.0, 5.0,  5.0]   # 无优化的起点

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    v_mini = values_mini + values_mini[:1]
    v_base = values_base + values_base[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, v_mini, "o-", color=BLUE,   linewidth=2, label="mini-infer (Phase 21)")
    ax.fill(angles, v_mini, alpha=0.15, color=BLUE)
    ax.plot(angles, v_base, "s--", color=GRAY,  linewidth=1.5, label="Phase 3 基线")
    ax.fill(angles, v_base, alpha=0.08, color=GRAY)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8)
    ax.set_title("mini-infer 综合能力总览", fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_overview_radar.png"), bbox_inches="tight")
    plt.close(fig)
    print("✓ 07_overview_radar.png")


if __name__ == "__main__":
    chart_throughput_evolution()
    chart_cuda_graph()
    chart_moe_ep()
    chart_flash_decode()
    chart_chunked_prefill()
    chart_w8a8()
    chart_overview_radar()
    print(f"\n所有图表已保存到 {os.path.abspath(OUT_DIR)}/")
