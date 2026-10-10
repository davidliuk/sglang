# SGLang：稀疏 min_new_tokens 惩罚优化

2026-10-10，已实现、提交并在 Colab A100 完成验证。核心收益集中在启用 `min_new_tokens` 的大 batch decode：最终提交下 128 并发吞吐 **+2.94%**，512 并发 **+6.53%**。单请求、真实 prefix hit 和 TTFT 没有稳定收益，负结果完整保留。

## 贡献选择与代码

选择 `BatchedMinNewTokensPenalizer`，其 stop/EOS 列表通常只有少量 token，原实现却准备 batch × vocabulary 的 FP32 惩罚表，每个 decode step 再做 dense `where` 和加法。这是从固定基线源码推导出的优化机会，收益方向随后由实测确认。[上游基线源码](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/sampling/penaltylib/min_new_tokens.py)。

筛选时也看了 [Unified LRU #24072](https://github.com/sgl-project/sglang/issues/24072)（已关闭、有相关实现）、[nonzero 同步 #9889](https://github.com/sgl-project/sglang/issues/9889)（需要按现有调用路径重新 profile）和 [SM120 #19637](https://github.com/sgl-project/sglang/issues/19637)（本次 A100 无法验证对应硬件优化）。最终选这个改动较小、每次 decode 重复访问大词表、可在现有 GPU 直接验证的点。

历史 [PR #28397](https://github.com/sgl-project/sglang/pull/28397) 已去掉该惩罚器的逐步 host sync；本次固定基线已包含它，优化的是仍然保留的 dense vocabulary 运算。文件最近一次功能提交是 [#39777 pinned metadata transfers](https://github.com/sgl-project/sglang/commit/f31a7bd45c6ab86796aa012ebfd2378bc42e1a59)，本次继续保留 pinned/nonblocking 传输。搜索快照见 `overlap-search.json`、`path-history.json`；搜索不保证没有尚未索引的并行工作。

- 工作目录：`python/sglang/srt/sampling/penaltylib/min_new_tokens.py` in the PR checkout
- 分支：`perf/sparse-min-new-tokens`
- baseline SHA：`234ccba7f625e12639e621044496bcc6197c3758`
- 修改后 SHA：`fcdd5e12ac1865c94f73807a9dc5370518608e0c`
- Implementation is the parent commit `fcdd5e12ac1865c94f73807a9dc5370518608e0c` of this evidence commit.
- English summary and validation tables: [README.md](README.md)

实现只保留 padded stop ID 与 compact penalty：CUDA 用单个 Triton kernel 更新尚未达到最小长度请求的 stop logits；其他设备走 Torch `scatter_add_`。保持原来的 EOS 来源合并、按生成步累计长度、filter/merge、资源清理和 additive 非有限数语义。CPU prepare 拒绝越界 ID，padding 不写入真实 token 0，合并不同 stop-list 宽度时补零；CUDA graph replay 读取最新长度计数。

```diff
 prepare
-  FP32 penalties[batch, vocabulary]
+  stop_ids[batch, max_stops] + penalties[batch, max_stops]
 apply each decode step
-  dense where + dense vocabulary add
+  one CUDA kernel touching active stop entries
```

例如 batch=512、vocab=151936、每行两个 stop token，仅该惩罚器保存的表从约 **296.75 MiB** 变为约 **12 KiB**（ID int64 + penalty FP32），并去掉它的 dense where 临时表。这是按 tensor 大小计算的局部结果；overlap sampling 的其他 dense accumulator、模型/KV 显存仍然存在，没有测得整个服务等比例节省显存。

## 环境与参数

| 项目 | 固定值 |
|---|---|
| GPU | 单卡 NVIDIA A100-SXM4-40GB |
| 系统 | Linux 6.6.122+；Python 3.12.3 |
| Driver / CUDA | 580.82.07 / 13.0 |
| Torch / Triton | 2.14.1+cu130 / 3.8.0 |
| Transformers / FlashInfer | 5.19.0 / 0.7.0.post1 |
| sglang-kernel | 0.4.9 |
| 模型 | Qwen/Qwen3-0.6B，BF16，TP=1 |
| 模型 revision | c1899de289a04d12100db370d81485cdf75e47ca |
| attention / sampling | triton / pytorch |
| context / chunked prefill | 4096 / 8192 |
| seed / stream interval | 42 / 1 |
| CUDA graph | decode 开启；prefill graph 关闭 |
| scheduler | 主实验 overlap 开启 |

默认组：max-running=128、max-total-tokens=32768、decode graph batch=1/8/32/128。512 扩展组：max-running=512、max-total-tokens=98304、graph batch 追加 512，且两侧都显式 `--disable-radix-cache`。其他参数相同。各组内部 baseline/修改版保持相同模型、输入、dtype、参数；不可直接把不同组绝对吞吐当作仅改变 batch 的结果。完整 server args/config 见 `results/final/*-info.json` 和 `results/high/*-info.json`；完整依赖见 [freeze.txt](results/freeze.txt)。

模型和依赖由 Colab 直接从 Hugging Face、官方 wheel/PyPI、GitHub 获取，本机仅传小脚本并取回结果。首次准备总计 **75.27 秒**：uv 3.97、venv 0.27、Torch 24.66、其他依赖 11.87、源码 8.29、模型下载 5.82、首次 import 20.38 秒。这是本次观测，不是下载性能保证。随后修正了旧 `datasets==1.1.1` 与 pyarrow 的导入冲突，最终 datasets=5.1.0；失败日志保留。每个服务 ready（包括加载/JIT/graph capture）约 34–40 秒，随后单独预热，再开始计时，未把准备/加载算进吞吐。

## 计时与输入

ABBA 顺序：baseline → 修改版 → 修改版 → baseline，每块独立服务进程，每场景 5 个计时 group。因此每场景每实现 **10 个样本**；不是把一个 batch 内数百个响应当作独立性能样本。表中为各实现 group 吞吐中位数的比值，单次单卡估计，未声称统计显著性或跨硬件通用收益。

吞吐 = 一组所有返回的 completion token 数 / 客户端发请求至收到完整响应的 wall time，包含 prefill、decode、HTTP、反序列化/SSE 消费。SSE TTFT = 客户端观察到首个含生成 token 事件的时间；batch 内先求各请求 TTFT 平均，再对 group 取中位数。非 SSE 组不报告 TTFT。SSE 和非 SSE 的吞吐不能直接混比。

输入为关闭 thinking 的 chat template，提示要求解释树为何需要水并列十项事实，约 31–33 prompt tokens，temperature=0、max_new_tokens=64；常规场景 min_new_tokens=64。mixed32 混合 0/8/32/64 最小长度、不同自定义 stop IDs、frequency/presence penalties。control 组 min_new_tokens=0。

cold 组每次计时前 flush，逐响应断言 cached_tokens=0。真实 cache 组先做 1 token 请求建立前缀，再计时，断言每个响应 cached_tokens>0，本次最小为 30。512 初次 warmup 出现批内前缀复用，未进入有效计时；保留 `high-cache-attempt/`，随后两侧禁用 radix cache 重跑。

**所有主计时响应都以 length=64 结束**，没有设置 ignore_eos=true。另用“2+2 只答数字”的 4 个短请求，min_new_tokens=0/1/4/8，验证实际 EOS 正常停止；这 32 个自然停止响应属于正确性验证，没有混入长输出吞吐。自然 EOS 下的性能收益尚未量化。

## 最终提交：真实模型结果

默认配置，140 个计时 group / 7,080 响应：

| 场景 | baseline tokens/s | 修改后 tokens/s | 变化 |
|---|---:|---:|---:|
| single | 366.29 | 362.53 | -1.03% |
| batch32 | 8,183.51 | 8,275.33 | +1.12% |
| batch128 | 17,094.84 | 17,596.94 | +2.94% |
| mixed32 | 8,013.61 | 8,104.37 | +1.13% |
| control128 | 18,448.63 | 18,271.69 | -0.96% |
| single / SSE | 352.47 | 345.80 | -1.89% |
| batch32 / SSE / prefix hit | 7,803.15 | 7,716.33 | -1.11% |

SSE TTFT（越低越好）：

| 场景 | baseline TTFT ms | 修改后 TTFT ms | 变化（越低越好） |
|---|---:|---:|---:|
| single | 43.09 | 43.64 | +1.29% |
| batch32 / prefix hit | 69.49 | 70.47 | +1.41% |


512 扩展，40 个计时 group / 20,480 响应：

| 场景 | baseline tokens/s | 修改后 tokens/s | 变化 |
|---|---:|---:|---:|
| batch512 | 27,626.55 | 29,429.45 | +6.53% |
| control512 | 31,491.03 | 31,657.26 | +0.53% |

原始负结果需要与正结果一起解释：最终默认组单请求、cache/SSE 有约 1–2% 退化，min=0 控制组也退化约 0.96%；512 控制组变化 +0.53%。这提示服务块之间存在其他变化，但仅靠控制组不能断言所有负值都是噪声，也不能替代原始结果。可以支持“大 batch 的 min-token 场景有收益”，不能支持“普遍改善 TTFT/单请求”。

### 同进程诊断（与主实验分开）

用一个常驻模型进程同时维护原/新惩罚器的 metadata 和 counters，通过 request custom_params 选择 apply 实现。双方都支付相同双份 metadata 开销。4 个 cycle，每个 cycle ABBA，两场景共 32 groups / 2,064 responses，输出完全一致。这是额外插桩控制，不能代替无插桩主实验。

| 场景 | baseline tokens/s | 修改后 tokens/s | 变化 |
|---|---:|---:|---:|
| single / SSE | 351.69 | 351.99 | +0.09% |
| batch128 | 16,916.22 | 17,604.26 | +4.07% |

单请求 SSE TTFT 44.97 → 44.58 ms（约 -0.86%），吞吐 +0.09%；batch128 吞吐 +4.07%。它支持“大 batch decode apply 是收益来源，单请求并无明显净收益”的解释；未定位其他默认 cache/SSE 的小幅负值，保留为未解决的泛化限制。诊断代码与审计记录在 `control-launcher.py`、`control.py`、`results/control/`。

## 算子测量

生产惩罚器真实 `apply()`，24 个形状：B=1/8/32/128/256/512 × V=32000/151936 × 自定义 stop 数 1/8（另加 EOS）；每形状 ABBA，每块 10 个 event sample，每 sample 连续 100 次 apply，共 960 event samples。此表 V=151936、一个 custom stop + 一个 EOS；各实现取 20 samples 中位数。

| Batch | baseline μs/apply | 修改后 μs/apply | 倍数 |
|---:|---:|---:|---:|
| 1 | 43.14 | 27.53 | 1.57× |
| 8 | 43.47 | 27.54 | 1.58× |
| 32 | 85.37 | 27.78 | 3.07× |
| 128 | 302.17 | 27.35 | 11.05× |
| 256 | 588.09 | 27.54 | 21.35× |
| 512 | 1161.36 | 27.42 | 42.35× |


CUDA event 包住 100 次主机发起的调用，包含可能的 launch 间隙，**不是纯 kernel profiler 时间**。另存 wall_us。这里只验证 apply 热路径，排除模型、请求准备、实际长度变化等开销；42× 不代表整个 SGLang 42×。完整全部形状/样本见 [micro-raw.json](results/micro-raw.json)。

## 正确性与检查

- 最终代码相关 pytest：**105 passed，23 subtests passed**，exit=0；主实验前后均运行。覆盖现有 penaltylib、sampling_batch_info、sampling_metadata_staging，以及新增 registered CUDA tests。
- CPU/dense oracle：ragged/empty stop list、token 0 padding、不同宽度 merge/filter、长度边界、无惩罚请求、EOS 来源去重、非法 ID、speculative repeat=3。
- CUDA：FP32/FP16/BF16/FP64、strided logits、empty/313 stops 跨 CTA、inf/nan 和 FP64 大有限数 additive 语义、CUDA graph replay；另有 **65 次 GPU dense differential**，24 个形状精确一致，graph counters 0/16/17/18/19/20 边界验证。
- 最终提交真实模型：**27,560 个计时响应 + 32 个自然 EOS 响应**，baseline/修改版 output_ids、text、finish_reason 全一致；自然 EOS 都为 stop=151645。诊断额外 2,064 个响应一致。
- `pre-commit` 全部通过，`git diff --check` 无错误，工作区干净；证据 [precommit.log](precommit.log)。

完整 pytest 命令（在准备好的 GPU 环境）：

```sh
PYTHONPATH=/content/sglang-src/python /content/sglang-env/bin/python -m pytest \
  /content/min-tokens/test_penaltylib.py \
  /content/min-tokens/test_min_new_tokens.py \
  /content/sglang-src/test/registered/unit/sampling/test_sampling_batch_info.py \
  /content/sglang-src/test/registered/unit/sampling/test_sampling_metadata_staging.py -q
```

测试副本 SHA 与提交内文件逐一核对。`final-tests.py` 会先还原 PR penalizer，测试由原源码 import；新增 kernel 在同一源码目录。完整日志 [final-tests.log](results/final-tests.log)。

## 探索阶段与未覆盖范围

最初 Torch eager compact scatter 版本在 batch1 43.29 → 52.34 μs，反而慢约 21%；Torch compile 版本 batch1 44.64 → 98.41 μs，也没有解决全部 dense 路径。两版都放弃，保留代码与失败/中断实验于 `prototype/`、`compiled-prototype/`。不能把这些不完整 e2e 数据合并为正式 ABBA。

最终加入 CPU 非法 ID guard 之前，曾跑完整 overlap/nonoverlap ABBA：280 groups / 14,160 responses，输出一致；其默认 overlap batch128 +4.26%、nonoverlap +3.43%。源码/内核 pre_guard 快照、原始样本均保留；它是**之前实现的探索结果**，不冒充最终 HEAD 验证。完整表如下：

| 场景 | baseline tokens/s | 修改后 tokens/s | 变化 |
|---|---:|---:|---:|
| single | 367.82 | 371.79 | +1.08% |
| batch32 | 8,168.12 | 8,380.83 | +2.60% |
| batch128 | 17,036.05 | 17,762.21 | +4.26% |
| mixed32 | 8,009.12 | 8,179.27 | +2.12% |
| control128 | 18,464.25 | 18,520.98 | +0.31% |
| single / SSE | 348.93 | 361.42 | +3.58% |
| batch32 / SSE / prefix hit | 7,798.06 | 8,013.73 | +2.77% |

nonoverlap：

| 场景 | baseline tokens/s | 修改后 tokens/s | 变化 |
|---|---:|---:|---:|
| single | 268.90 | 268.89 | -0.00% |
| batch32 | 6,059.72 | 6,152.95 | +1.54% |
| batch128 | 13,541.98 | 14,006.84 | +3.43% |
| mixed32 | 5,909.28 | 5,988.06 | +1.33% |
| control128 | 14,278.39 | 14,241.26 | -0.26% |
| single / SSE | 262.88 | 263.12 | +0.09% |
| batch32 / SSE / prefix hit | 5,653.39 | 5,701.77 | +0.86% |

当前未覆盖：其他 GPU 架构/AMD/NPU、tensor parallel、多模型/量化、长 prefill、真实模型 speculative decoding、大规模动态 admission、全仓 CI 和自然短输出性能。CPU fallback 通过单元测试，其他设备未实机验证。加速局限于启用 min_new_tokens 的路径，min=0 控制组没有预期直接收益；主实验没有宣称提升默认全部请求。

## 复现与证据保存

最终源码快照：`pr_min_new_tokens.py`、`min_new_tokens_kernel.py`、两份测试，与 commit fcdd5e12 完全一致。

先准备小脚本，再分配独立 GPU session；在空 runtime 使用 `send-prepare.py`（约 100 KiB）解包脚本并后台运行 prepare，观察 worker.pid、prepare.log、READY.json，避免超时后重复启动。

```sh
# 示意，SESSION 使用本次新分配的专属 session 名
colab exec -s SESSION --file send-prepare.py
# 等 READY.json，确认 prepare worker 已结束
colab exec -s SESSION --file start-experiment.py
# 等 final/DONE.json；失败查 FAILED.json/log，不能立即重复启动
# 以下在 VM 的 venv 下依次执行，前一 worker 完成后再执行：
PYTHONPATH=/content/sglang-src/python /content/sglang-env/bin/python /content/min-tokens/run-high.py
PYTHONPATH=/content/sglang-src/python /content/sglang-env/bin/python /content/min-tokens/control.py
/content/sglang-env/bin/python /content/min-tokens/collect.py
```

prepare 固定源码与模型 revision，从上游直接下载；最终依赖见 requirements/freeze。`start-experiment.py` 安装新增 kernel，`run-final.py` 运行测试、算子、GPU differential、默认 e2e，再还原 PR 并重测；`run-high.py` 完成 512 扩展并还原 PR。Colab CLI 连接中断时先查 PID/DONE/FAILED，必要时使用该 session 的 SSH 代理读取，不能重复创建 worker。生产测试与 e2e 脚本本次均执行过；更新后的组合 bootstrap 做过语法/嵌入一致性检查，未另起全新 VM 重放整个入口。

[results.tar.gz](results.tar.gz) 大小 4,395,420 bytes，远端/本地 SHA256 同为：

`b36c8a021941b75fcf4454b4987649382e2cdc27a320d56b5e319f7023449b0b`

不包含模型权重。所有原始输出、单样本计时、服务配置、worker logs、失败探索、freeze、硬件数据均已取回。运行 `python3 verify-results.py` 可离线重新核对 archive/source hashes、输出与 finish parity、样本数并生成 [summary.json](summary.json)；无需 CUDA。

只停止本任务的 `sglang-min-tokens-20261010`，取回/hash 核对后停止，最终 sessions 显示无 active sessions，usage 显示 active assignments=0、usage rate=0。compute balance 1025.17 → 1021.49，约 **3.68 compute units**；该余额差含本次准备、失败探索、验证与等待。记录 [sessions-after.txt](sessions-after.txt)、[usage-after.txt](usage-after.txt)。本机临时 SSH key 已清理。
