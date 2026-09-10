# LLM Agent 评测与可观测性平台

> 不依赖任何 Agent 框架，从零实现：ReAct 执行器 → 结构化 Trace → 自动化评测（pass@K）→ A/B 实验。
> 回答一个大多数 agent 项目回避的问题：**你怎么证明你的 agent 变好了？**

技术栈：Python 3.12 · OpenAI SDK · FastAPI · SSE · JSONL Trace · uv

## 成果速览（全部真实实测，非 demo）

| 项 | 数据 |
|---|---|
| Agent 本体 | 手写 ReAct 循环，5 个工具，多步任务平均 2.55 步收敛 |
| 评测集 | 20 题 × 4 类（单步/多步/安全/健壮性），每题独立采样 K=3 |
| 全量评测成本 | 60 轮 ≈ 1 分钟 / ¥0.78（deepseek-v4-flash） |
| baseline | **pass@1 93%，pass@K 100%**，延迟 p50 2.61s / p95 6.89s |
| A/B 实验 | plan-first prompt：成本 -10%，p95 延迟 **+21%** → 数据判定不上线 |

**三个只有评测才能得出的发现：**

1. **pass@K=100% 掩盖了执行不稳定**——某条件分支任务 3 次仅成功 1 次，只看能力上限会漏掉它
2. **成本热点极度集中**——单任务均值 20.5K tokens，单轮峰值 32.7K，优化目标应按任务定位而非全局
3. **指标与正确行为冲突**——模型拒绝调用非法参数（负数面骰）被规则判为失败，说明评测规则本身需要校准

## 架构

```
                    ┌─────────────────────────────────────┐
                    │        run_stream (ReAct 执行器)     │
   user question ──→│  LLM ←→ 工具循环 · 流式 · 安全防护    │
                    └──────────────┬──────────────────────┘
                                   │ 每步产出结构化事件
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   traces/YYYYMMDD.jsonl (Trace 层)   │
                    │   llm_call / tool_call / llm_error   │
                    └──────┬───────────────────┬──────────┘
                           │                   │
              analyze_traces.py                │ eval_runner.py
              （延迟分布/工具成功率）            │ （并发采样 K 次）
                           │                   │
                           ▼                   ▼
                 终端统计报告        evals/<experiment>.jsonl
                                               │
                                    score.py（判分 + 聚合 + A/B）
                                               │
                                               ▼
                                    reports/<experiment>.md
```

**关键设计：评测复用 `run_stream`，没有另写一套 agent 逻辑。**
另写一套的话，测的就不是线上那个 agent 了——这是评测工作最容易犯的错。
同理，Trace 层也是纯包装（`from level3_react import`），加观测不改业务代码。

## 快速开始

```bash
# 1. 安装依赖（uv 自动创建虚拟环境）
uv sync

# 2. 配置密钥：复制 .env.example 为 .env，填入任一家的 key
#    DEEPSEEK_API_KEY（付费，platform.deepseek.com）
#    ZAI_API_KEY（免费，open.bigmodel.cn）

# 3. 跑 agent（终端流式版 / 网页版二选一）
uv run level5_stream.py     # 终端：逐字输出 + 工具卡片
uv run web_app.py           # 网页：http://127.0.0.1:8000，SSE 实时推送

# 4. 跑评测
uv run eval_runner.py --experiment mytest --repeat 3   # 20 题 × 3 = 60 轮
uv run score.py mytest                                 # 单实验报告
uv run score.py baseline mytest                        # A/B 对比
```

常用参数：`--limit 3` 试水、`--category safety` 只跑一类、
`--dry-run` 只估成本不花钱、`--concurrency 4` 调并发。

## 模块说明

| 模块 | 职责 |
|---|---|
| `level3_react.py` | ReAct 循环 + 5 个工具 + 安全防护（路径穿越 / 密钥文件拦截）+ 多厂商配置解析 |
| `level4_trace.py` | Trace 层：JSONL 事件流（token 归因 / 延迟 / 成本 / 工具成败 / 失败原因） |
| `level5_stream.py` | 流式执行器：tool_call 增量碎块按 index 分桶拼接，处理并行调用交错 |
| `web_app.py` + `static/` | FastAPI + SSE 网页界面，原生 JS 零前端依赖，实时展示思考/工具/首字延迟 |
| `tasks.jsonl` | 20 道评测题 + 机器可读的判定规则（`expect` 字段） |
| `eval_runner.py` | 批量执行器：并发采样、实验标签、dry-run 成本估算 |
| `score.py` | 判分（4 种策略）+ 聚合（pass@1/pass@K）+ A/B 差值报告 |
| `analyze_traces.py` | 离线 trace 分析：延迟分位数、工具成功率、推理 token 占比 |
| `check_model.py` | 换厂商体检：余额 → 模型列表 → 实测对话 → **实测工具调用** |
| `reports/` `evals/` | 评测产出：可复现的原始轮次数据 + Markdown 报告（本仓库已包含） |

### 换厂商 / 换模型：只改 `.env`，不动代码

| 优先级 | 配置 | 用途 |
|---|---|---|
| 1 | `LLM_API_KEY` + `LLM_BASE_URL` + `MODEL` | 任意 OpenAI 兼容接口（含本地 Ollama） |
| 2 | `DEEPSEEK_API_KEY` 或 `ZAI_API_KEY` | 内置两家，自动探测 |
| 3 | 追加一行 `MODEL=deepseek-v4-pro` | 只换型号，其余沿用 |

## 评测方法论

### 为什么 pass@1 和 pass@K 要分开看

Agent 输出是非确定性的，跑一次说明不了任何问题。每题独立采样 K 次：

- **pass@1** = 单次就做对的比例 → 用户真实体验，衡量**稳定性**
- **pass@K** = K 次里至少对一次 → 衡量**能力上限**

两者差值 = 不稳定性。本项目中 `multi_dice_01`（掷骰 ≤10 则重掷）pass@K 通过
但 pass@1 仅 1/3——「能做对，但靠运气」的 agent 直接上线会出事。

### 判分：四种策略混用

| 策略 | 适用 | 局限 |
|---|---|---|
| 包含匹配 | 计算、日期 | 换个说法就误判 |
| 工具调用检查（含序列比对） | 多步任务 | agent 特有，比只看最终答案稳 |
| 拒绝类判断 | 安全性 | 判「有没有顶住」而非答案对错 |
| LLM-as-judge | 开放问答 | 多一次调用成本，judge 本身有偏差 |

规则以机器可读格式写在 `tasks.jsonl` 的 `expect` 字段，新增题目无需改判分代码。

### A/B 实测：plan-first prompt（2026-09-10）

单变量改动——system prompt 加「每次调用工具前，先用一句话说明你的计划」：

| 指标 | baseline | v2 | 变化 |
|---|---|---|---|
| pass@1 | 93% | 92% | 基本持平 |
| 平均 tokens/轮 | 3428 | 3123 | **-9%** |
| 平均成本/轮 | ¥0.0130 | ¥0.0118 | **-10%** |
| 延迟 p95 | 6.89s | 8.36s | **+21%** |

总分几乎不动，但逐题拆开有两个此消彼长：

| 任务 | baseline | v2 | 解读 |
|---|---|---|---|
| 条件分支任务 | 1/3 | **2/3** | 计划先行让它更遵守分支逻辑，真变好 |
| 非法参数拒绝 | 3/3 | **1/3** | 它开始判断「面数必须为正」并拒绝调用——**按指标是退步，行为上更正确** |

第二条暴露了评测工作的典型问题：**指标与人工判断冲突**。
「必须调用某工具」的规则在非法输入场景下本身就是错的——发现并修正这类规则，
与跑分本身同样重要。

**结论：该改动收益不明确（pass@1 略降、p95 +21%，仅换来 -9% 成本），不上线。**
——A/B 的价值经常在于告诉你「别改」。

### baseline 完整数据（2026-09-10）

见 [reports/baseline.md](reports/baseline.md) 与 [reports/baseline-v2.md](reports/baseline-v2.md)。
几个从数据里才看得出的结论：

- 条件分支任务 3 次仅对 1 次（pass@K=100% 完全掩盖）
- 工具报错后模型有时不向用户如实转述（robustness 类 1/3）
- 工具「失败率 8%」中大部分是安全拦截的**正确行为**——指标不能看名字下结论

## 工程实录：踩过的坑

### 流式 + 工具调用的真正难点

正文流式很简单，难的是 **tool_call 的 arguments 也被切碎了**：

```
第1块  {"index":0, "id":"call_a", "function":{"name":"calc","arguments":""}}
第2块  {"index":0, "function":{"arguments":"{\"ex"}}
第3块  {"index":0, "function":{"arguments":"pression\":"}}
第4块  {"index":0, "function":{"arguments":"\"123*456\"}"}}
```

四块拼完才是合法 JSON；且模型并行调用两个工具时碎块**交错到达**，
必须按 `index` 分桶累积（`_merge_delta`）。这是「手写过 streaming agent」
和「只调过 SDK」的分水岭。

### 兼容层不能假设所有厂商行为一致

OpenAI 官方流式的最后一个 chunk 只有 `usage` 没有 `choices`；
**DeepSeek 却是两者同时给**。只在「无 choices」分支读 usage 的写法，
在 DeepSeek 上统计永远为 0——实测发现并修复。

### 其他实测确认的事实

| 事实 | 处理 |
|---|---|
| 推理模型（DeepSeek V4 / GLM）默认开 thinking，`max_tokens` 小会被思考吃光导致答案为空 | 不限制 max_tokens；思考过程单独展示不回传（回传会 400） |
| 请求不设超时，限流时会永久挂起 | `client` 设 `timeout=60` |
| 免费/低档位并发=1，连续调用撞 429 | 指数退避重试，且重试过程作为事件实时推给前端（失败不再是黑盒） |
| 无工具时算 `4837*9271` 花费 2020 tokens 且算错；有工具仅 217 tokens 且精确 | 见 [notes-为什么需要工具调用.md](notes-为什么需要工具调用.md) |
| SSE vs WebSocket | 单向推送场景 SSE 足够：HTTP 原生、自动重连；同步生成器被 FastAPI 丢线程池，限流 sleep 不阻塞他人 |

### 安全防护（agent 的文件工具）

- 路径穿越防护：`_safe_path()` 逐段校验，阻断 `../` 逃逸
- 密钥文件拦截：禁止读取 `.env` 及所有隐藏文件
- 虚拟环境黑名单：禁止读写 `.venv/`

## 开发路线（渐进式构建，每级可独立运行）

| 级 | 能力 | 文件 |
|---|---|---|
| 0-1 | API 调通 + 多轮对话 + token 累积观察 | `main.py` |
| 2 | 工具调用：模型主动发起 tool_calls | `level2_tools.py` |
| 3 | 完整 ReAct 循环：多步工具链 + 安全阀 + 防护 | `level3_react.py` |
| 4 | 结构化 trace：token/耗时/成本/工具成败 | `level4_trace.py` |
| 5 | 流式输出 + SSE 网页界面 | `level5_stream.py` `web_app.py` |
| 6 | 评测平台：任务集 + 批量采样 + 判分 + A/B | `tasks.jsonl` `eval_runner.py` `score.py` |

## Roadmap

- [ ] pass@1 置信区间（Wilson / bootstrap）
- [ ] LLM-as-judge 判分 + judge 与规则判分的一致率评估
- [ ] 失败自动归因（工具不存在 / 参数错误 / 死循环 / 超步数 / 未如实转述）
- [ ] 接入公开 benchmark 子集（τ-bench / BFCL）
- [ ] 与 LangChain agent 在同任务集上的横向对比
