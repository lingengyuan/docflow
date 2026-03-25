# Embedding Runtime Experiments

> Date: 2026-03-25
>
> Status: completed

## 1. 背景

当前 DocFlow 的 ingest 主瓶颈已经比较明确：**大文件导入时，主要耗时在 embedding，不在 parse / chunk**。

在不明显降低检索质量的前提下，这一轮的目标是验证：

1. 本地 ONNX 是否能替代当前 `torch + CPU` 路径。
2. 外部 embedding runtime（先试 TEI）是否能在 **不改业务代码** 的前提下明显加速。


## 2. 测试环境

- 机器：M5 MacBook Air，32 GB 统一内存，1 TB SSD
- 模型：`Qwen/Qwen3-Embedding-0.6B`
- 当前生产基线：`sentence-transformers` + `torch` + `CPU`
- 真实样本来源：
  - `/Users/hughlin/MyNotes/HughLin/Notes/plans/plans/intelligent-ops/intelligent-ops-automation-plan-v2.md`
- 样本规模：
  - ONNX 对比：同一份大 Markdown 的前 `128` 个真实 chunks
  - TEI 对比：同一份大 Markdown 的前 `64` 个真实 chunks


## 3. 实验 A：本地 ONNX backend

### 3.1 目标

验证本地 ONNX 导出是否能在当前硬件上跑赢 `torch CPU`，并且保证 ingest/query 走同一套 embedding runtime 配置，避免向量空间漂移。


### 3.2 做了什么

- 新增共享 backend 层：`src/embedding_backend.py`
- ingest 与 query 统一走同一套 embedding backend 配置
- embedding cache key 纳入 backend 维度，避免 `torch / onnx` 混用污染缓存
- 修复 Qwen3 ONNX 的真实兼容性问题：
  - 本地导出后推理会要求 `position_ids`
  - 在共享 backend 中补齐 `position_ids` 注入


### 3.3 验证结果

- `PYTHONPATH=. .venv/bin/pytest tests/ -q` → `60 passed`
- ONNX smoke test 通过：
  - `hello world` 可正常返回 `1024` 维向量


### 3.4 基准结果

同一批 `128` 个真实 chunks：

| Runtime | Total | sec/chunk |
|---|---:|---:|
| `torch CPU` | `34.388s` | `0.2687` |
| `ONNX CPU` | `73.892s` | `0.5773` |


### 3.5 结论

- ONNX 路径已经**功能可用**
- 但在当前这台 M5 + 这个模型组合上，**明显慢于现有 `torch CPU`**
- 因此默认配置已保持 `embedding.backend: "torch"`
- ONNX 代码保留为实验入口，不作为当前默认提速方案


## 4. 实验 B：TEI sidecar（不改 repo 代码）

### 4.1 目标

在**不修改项目源码**的前提下，验证独立 embedding runtime 是否能作为提速方向。


### 4.2 约束

- 不改仓库代码
- 只使用临时环境
- 实验完成后清理临时安装、日志、缓存


### 4.3 尝试 1：Docker TEI

使用思路：

```bash
docker run ghcr.io/huggingface/text-embeddings-inference:cpu-latest ...
```

结果：

- 在本次测试使用的 tag 上，Apple Silicon 命中了平台问题
- Docker 返回：没有匹配的 `linux/arm64/v8` manifest

结论：

- 这条 Docker 快速验证路径本轮未跑通


### 4.4 尝试 2：Homebrew 原生安装 TEI

做法：

- 使用 Homebrew 临时安装 `text-embeddings-inference`
- 启动：

```bash
text-embeddings-router \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --port 8091 \
  --max-client-batch-size 128
```

观察到的关键信息：

- TEI 在这台机器上实际选择了 **Metal** backend
- 模型能够正常启动并对外提供 `/embed`


### 4.5 TEI 与当前基线对比

测试样本：同一份大 Markdown 的前 `64` 个真实 chunks

| Runtime | Total | sec/chunk | 说明 |
|---|---:|---:|---|
| 当前基线 `torch CPU` | `34.187s` | `0.5342` | DocFlow 当前路径 |
| TEI（Metal，首次完整请求） | `142.780s` | `2.2309` | 已完成 ready + warmup |
| TEI（Metal，重复请求） | `120.295s` | `1.8796` | 比首轮略好，但仍明显更慢 |


### 4.6 向量一致性检查

对前 `8` 条样本，把 TEI 与当前本地 `torch` 结果做归一化后 cosine 对比：

- cosine mean: `0.999996`
- cosine min: `0.999995`

解释：

- **向量质量 / 一致性没有看到明显问题**
- 当前 blocker 是**速度明显更慢**，不是向量漂移


### 4.7 `float16` 额外尝试

追加尝试：

```bash
text-embeddings-router \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --port 8091 \
  --max-client-batch-size 128 \
  --max-batch-tokens 32768 \
  --dtype float16
```

结果：

- 本轮没有拿到稳定的有效 benchmark
- `/embed` 请求出现 `502`
- 进程未保持稳定可用状态

结论：

- 在当前机器 / 当前模型 / 当前版本组合下，`float16` 这条参数分支**不稳定**


## 5. 实验 C：Infinity 临时探测（未完成 benchmark）

这轮还做过一个非常轻量的 backup probe：

- 在 `/tmp` 下创建临时 venv
- 只尝试安装 `infinity-emb`
- 不接入项目代码

碰到的问题：

- `infinity-emb 0.0.77` 在当前临时环境里拉到了较新的 `optimum` / `transformers`
- CLI 还会引用 `optimum.bettertransformer`
- 继续回退依赖时，出现较重的版本回溯和兼容性拉扯

结论：

- 本轮没有产出可比较的 Infinity benchmark
- 如果后续要认真测 Infinity，建议直接用**单独 pinned 环境或容器**，不要临时在混合环境里硬拼


## 6. 总结结论

### 当前可确定的结论

1. **当前默认 `torch CPU` 仍是最合理的生产选择**
   - 本地 ONNX：功能可用，但更慢
   - TEI（Metal）：向量一致性很好，但速度明显更慢

2. **ONNX 值得保留为实验入口，但不值得切为默认**

3. **TEI 暂时不值得优先集成**
   - 至少对当前 M5 + `Qwen3-Embedding-0.6B` 这组组合，不是收益方向


### 如果未来还要继续追 embedding 加速

建议优先级：

1. `Infinity` 的独立 PoC（单独容器或 pinned 环境）
2. 继续关注 Apple Silicon 上更适合 Qwen3 的 runtime
3. 再之后才考虑继续投入 P2 级 ingest 微优化


## 7. 收尾说明

- TEI 与 Infinity 这轮 PoC 都**没有修改仓库源码**
- 临时安装、缓存、日志、样本文件均已清理
- 仓库工作区保持干净
