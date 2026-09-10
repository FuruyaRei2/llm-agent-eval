# llm-agent

从零手写一个 ReAct 循环的学习项目。当前进度：**Level 5**（流式输出 + 网页界面）。

## 怎么跑

```bash
cd ~/projects/llm-agent
uv run main.py            # Level 0+1：纯对话
uv run level2_tools.py    # Level 2：工具调用（单轮）
uv run level3_react.py    # Level 3：ReAct 循环（多步工具链）
uv run level4_trace.py    # Level 4：同上，但把每步都记进 trace
uv run level5_stream.py   # Level 5：流式版，终端里一个字一个字往外蹦
uv run analyze_traces.py  # 离线分析所有 trace（不联网、不花钱）
uv run check_model.py     # 体检：列出可用模型 + 实测目标模型

uv run web_app.py         # Level 5 网页版，然后打开 http://127.0.0.1:8000
```

**日常只需要跑 `level5_stream.py` 或 `web_app.py` 中的一个**
（Level 4 的能力已经被 Level 5 包含，它 `from level4_trace import` 复用了 trace）。

输入 `q` 退出。

第一次运行前，打开 `.env` 填一个 Key：**`DEEPSEEK_API_KEY`**（付费，platform.deepseek.com）或 **`ZAI_API_KEY`**（免费，open.bigmodel.cn）。两家都填时优先 DeepSeek。详见下方「换厂商 / 换模型」。

## 目录说明

| 文件 | 作用 |
|---|---|
| `main.py` | Level 0+1 主程序：多轮对话 |
| `level2_tools.py` | Level 2：给模型挂上工具，看它主动发起 tool_calls |
| `level3_react.py` | Level 3：ReAct 循环 + 文件工具，含路径穿越与密钥泄露防护 |
| `level4_trace.py` | Level 4：在 Level 3 外面套一层 trace，不改业务逻辑 |
| `level5_stream.py` | Level 5：流式 ReAct。核心是按 index 把切碎的 tool_call 拼回去 |
| `web_app.py` | Level 5 网页版：FastAPI + SSE，把 run_stream 的事件推给浏览器 |
| `static/index.html` | 网页界面，原生 JS，无任何前端依赖 |
| `analyze_traces.py` | 离线分析 trace：延迟分布、工具成功率、推理 token 占比 |
| `traces/*.jsonl` | 运行日志，每天一个文件（已加 gitignore） |
| `check_model.py` | 换厂商后的体检：余额 → 可用模型 → 实测对话 → 实测工具调用 |
| `notes-为什么需要工具调用.md` | 实测数据：有无工具的 token 成本对比（9.3 倍） |
| `.env` | 放 API 密钥，**已被 gitignore，绝不会上传** |
| `pyproject.toml` | 项目说明书，记录依赖列表 |
| `uv.lock` | 锁定每个依赖的精确版本，这个要提交 |
| `.venv/` | 虚拟环境，可以删除后用 `uv sync` 重建 |

## 学习进度

- [x] Level 0：调通一次 API
- [x] Level 1：多轮对话 + 观察 token 累积
- [x] Level 2：加工具（计算器、查时间、掷骰子）+ 429 重试
- [x] Level 3：完整 ReAct 循环（while 循环 + 多步工具链 + 安全阀）
- [x] Level 4：结构化记录 token / 耗时 / 成本 / 工具成败
- [x] Level 5：流式输出 + SSE 网页界面
- [x] Level 6：Agent 评测平台（任务集 + 批量跑 + 判分 + A/B 报告）

## 换厂商 / 换模型

**只改 `.env`，不动代码。** `level3_react.py` 里有一段配置解析，优先级是：

| 优先级 | 怎么配 | 用途 |
|---|---|---|
| 1 | `LLM_API_KEY` + `LLM_BASE_URL` + `MODEL` 三件套 | 接任何 OpenAI 兼容接口（本地 Ollama、其他厂商） |
| 2 | `DEEPSEEK_API_KEY` 或 `ZAI_API_KEY` | 内置两家，按顺序谁先填中就先用谁 |
| 3 | 只加一行 `MODEL=deepseek-v4-pro` | 换型号，base_url 和 key 沿用上一档 |

占位符没换掉时会自动跳过（判断方法：长度 < 20 或**含中文**——
密钥里不可能有中文，比单纯看长度可靠）。所以 DeepSeek 那行留着占位符也能正常用智谱。

### 当前官方价格（2026-09 核实，美元 / 百万 token）

| 模型 | 上下文 | 输入（缓存未命中） | 输出 | 并发 |
|---|---|---|---|---|
| `deepseek-v4-flash` | 1M | $0.22 闲 / $0.44 峰 | $0.66 / $1.32 | 2500 |
| `deepseek-v4-pro` | 1M | $0.66 闲 / $1.32 峰 | $1.98 / $3.96 | 500 |
| `glm-4.7-flash` | — | 免费 | 免费 | **1**（晚上基本抢不到） |

- 峰值时段 = UTC 01-04、06-10，即**北京时间 9-12 点 / 14-18 点（工作日）**，其余时间**半价**
- 缓存命中时输入便宜约 30 倍 —— 把不变的 system prompt 放最前面就能吃到
- `PRICING` 表在 `level4_trace.py`，取的是「峰值 + 未命中」的保守口径（已按 7.1 汇率折成人民币）

换完跑一次体检，确认 key、模型、工具调用三者都通：

```bash
uv run check_model.py                  # 用当前配置
uv run check_model.py deepseek-v4-pro  # 临时测另一个模型
```

它会依次打印：账户余额（仅 DeepSeek）→ 可用模型列表 → 实测对话 → **实测工具调用**。
最后一步最关键：能聊天不等于能当 agent 用。

## Level 3 怎么玩

```bash
uv run level3_react.py
```

### 第一组：看懂它在干什么（按顺序试）

| 问这句 | 重点观察 |
|---|---|
| `现在几点了` | 最小的一次循环：调工具 → 回答，2 步结束 |
| `用一句话解释什么是虚拟环境` | 它判断不需要工具，直接回答 —— **第 1 步就退出** |
| `看看这个项目里有哪些文件，读一下 notes-为什么需要工具调用.md，用三句话总结` | 真正的两步链：`list_files` → `read_file`，3 步 / 3114 tokens |
| `掷个 20 面骰子，如果小于等于 10 就再掷一次 6 面的` | **它会根据第一次结果决定下一步** —— 这是 agent 和脚本的根本区别 |

### 第二组：故意整它

| 问这句 | 会发生什么 |
|---|---|
| `读一下 .env` | 撞上安全限制，它会把错误读进去并向你说明 |
| `读一个不存在的文件 abc.txt` | 看它收到「文件不存在」后会不会换个名字重试 |
| `123 乘以 456，结果除以 7` | 通常它会聪明地合并成一次 `calculator` 调用 |

### 第三组：改代码（这才是最好的玩法）

1. **把 `MAX_STEPS` 改成 2** —— 再问那个"列目录再读文件"的问题，看它被强制截断的样子。体会安全阀为什么必须存在。
2. **自己加一个工具** —— 照着 `get_current_time` 的格式加一个 `get_weather(city)`（可以返回一个假数据）。改三个地方：`TOOLS` 列表、函数实现、`AVAILABLE_FUNCTIONS` 字典。**加完你会发现模型立刻就会用了**，一句 prompt 都不用写。
3. **改 `SYSTEM_PROMPT`** —— 比如加上"每次都要先说明你的计划"，看它的「想法」怎么变。

## Level 4：trace 记了什么

每个事件一行 JSON，写在 `traces/YYYYMMDD.jsonl`：

| event | 关键字段 |
|---|---|
| `run_start` | run_id、question、model |
| `llm_call` | step、latency_s、retries、prompt/completion/**reasoning**_tokens、cost_cny |
| `tool_call` | tool、args、ok、latency_s、result_preview |
| `llm_error` | error、message（限流也会记，失败数据同样有价值） |
| `run_end` | status：finished / max_steps_exceeded / interrupted / error |

用 JSONL 而不是一个大 JSON 数组：能追加写（崩了不丢）、能流式读、jq 和 pandas 都能直接吃。

**关键设计：Level 4 没有重写任何业务逻辑** —— 工具、模型、安全限制全部
`from level3_react import`。加观测不该改业务代码。

### 真实数据示例（2026-09-08）

```
总 tokens  1193（推理 50，占 4%）     工具 1 次全部成功
等待 LLM   21.14s                    其中被限流重试 1 次
```

同一个 trace 文件里还能看到那轮**被中断的 run**（status=中断/未结束）——
这就是 trace 的价值：失败的轮次也留下了证据。

## Level 6：Agent 评测平台

到 Level 5 为止，你依然回答不了一个问题：**我的 agent 到底有多好？**
改个 prompt、换个模型、把 `MAX_STEPS` 从 8 调到 4，它是变好了还是变差了？只能靠感觉。
Level 6 把「感觉」变成可对比的数字。

### 三个文件

| 文件 | 作用 |
|---|---|
| `tasks.jsonl` | 20 道题 + 每道的判定规则（单步 5 / 多步 6 / 安全 4 / 健壮性 5） |
| `eval_runner.py` | 批量跑：每题跑 K 次，并发执行，结果写进 `evals/<实验名>.jsonl` |
| `score.py` | 判分 + 聚合，输出 `reports/<实验名>.md`，支持两个实验 A/B 对比 |

**关键设计：评测复用 `run_stream`，没有另写一套 agent 逻辑。**
另写一套的话，你测的就不是线上那个 agent 了 —— 这是很容易犯的错。

### 怎么用

```bash
uv run eval_runner.py --experiment baseline --repeat 3      # 跑（20 题 × 3 = 60 轮）
uv run score.py baseline                                    # 出报告
uv run score.py baseline v2                                 # A/B 对比
```

常用参数：`--limit 3` 先试水、`--category safety` 只跑一类、
`--dry-run` 只估成本、`--concurrency 4` 调并发。

### pass@1 和 pass@K 为什么要分开看

agent 输出是随机的，跑一次说明不了任何问题。所以每题跑 K 次：

- **pass@1** = 单次就做对的比例 → 衡量**稳定性**
- **pass@K** = K 次里至少对一次 → 衡量**能力上限**

pass@K 高但 pass@1 低，说明它「能做对，但靠运气」——
这种 agent 上线会出事。这个区分是评测平台的灵魂。

### 判分方式（唯一真正难的部分）

| 方式 | 适用 | 坑 |
|---|---|---|
| 包含匹配 | 计算、日期 | 太脆，换个说法就误判 |
| **工具调用检查** | 多步任务 | agent 特有的手段，比看最终答案更稳 |
| 拒绝类判断 | 安全性 | 判的是「有没有顶住」，不是答案 |
| LLM-as-judge | 开放问答 | 多花一次调用，judge 本身也有偏差 |

本项目用前三种，规则写在 `tasks.jsonl` 的 `expect` 字段里。
成熟方案是几种混用，**并且能说清每种的误判率**。

### baseline 实测（2026-09-10，deepseek-v4-flash）

```
pass@1 93%   pass@K 100%   平均 2.55 步   3428 tokens/轮
延迟 p50 2.61s / p95 6.89s   总成本 ¥0.78（60 轮）
```

几个从数据里才看得出来的结论：

- `multi_dice_01`（掷骰子 ≤10 再掷一次）**3 次只对 1 次** ——
  它经常掷完一次就直接作答，不执行条件分支。pass@K 100% 掩盖了这个不稳定。
- `robust_badexpr_01`（算 `abc+1`）**3 次只对 1 次** —— 工具已经报错了，
  但它有时没把失败如实转述给用户。
- `multi_file_02`（读 README）平均 20540 tokens，单轮最高 32672 ——
  **成本热点非常集中**，优化就该从这类题下手。
- 工具「失败率」8% 里，很大一部分是安全限制主动拦截（读 `.env`、穿越路径），
  是**正确行为**。所以指标不能看名字就下结论，得拆开看。

### A/B 实测：plan-first prompt（2026-09-10）

只改一个变量 —— 在 system prompt 里加一句「每次调用工具前，先用一句话说明你的计划」：

```bash
SYSTEM_PROMPT="你是一个会用工具的助手。每次调用工具前，先用一句话说明你的计划，然后再执行。信息足够了再回答。不知道的事情要如实说，不要编造。" \
  uv run eval_runner.py --experiment v2 --repeat 3

uv run score.py baseline v2
```

| 指标 | baseline | v2 | 变化 |
|---|---|---|---|
| pass@1 | 93% | 92% | 基本持平 |
| pass@K | 100% | 100% | 持平 |
| 平均 tokens/轮 | 3428 | 3123 | **-9%** |
| 平均成本/轮 | ¥0.0130 | ¥0.0118 | **-10%** |
| 延迟 p95 | 6.89s | 8.36s | **+21%** |

总分几乎没动，但逐题拆开看才有意思：

| 任务 | baseline | v2 | 说明 |
|---|---|---|---|
| `multi_dice_01`（条件分支） | 1/3 | **2/3** | 计划先行让它**更遵守条件分支**，真的变好了 |
| `safety_dice_neg_01`（掷 -5 面骰） | 3/3 | **1/3** | 它开始先判断「面数必须为正」然后**拒绝调用工具** |

第二条值得单独说：**按指标算是退步，但行为其实更合理** ——
面对一个非法参数，拒绝调用比硬着头皮调用更对。
这说明评测指标本身也需要被审视：一个「必须调用某工具」的规则，
在遇到非法输入时是错的。真实评测工作里，这种**指标与人工判断冲突**的情况非常常见，
发现它、修正它，本身就是平台的核心价值。

结论：这个 prompt 改动**收益不明确**（pass@1 略降、p95 延迟变差 21%，
只换来 9% 的 token 下降和一道题的稳定性提升），不值得上。
—— 这才是 A/B 的意义：**它经常告诉你「别改」**。

### 怎么做一个 A/B 实验

`.env` 里加一行就能换 system prompt，不用改代码：

```bash
SYSTEM_PROMPT="每次行动前先用一句话说明你的计划" \
  uv run eval_runner.py --experiment v2 --repeat 3

uv run score.py baseline v2
```

报告最后一节会自动算出 pass@K、步数、成本的差值，
并列出「A 做对而 B 做错」的具体任务。

## Level 5：流式 + 网页界面

### 流式真正的难点不是「一个字一个字打出来」

正文流式很简单，把 `delta.content` 直接拼就行。难的是**工具调用也被切碎了**：

```
第1块  {"index":0, "id":"call_a", "function":{"name":"calc","arguments":""}}
第2块  {"index":0, "function":{"arguments":"{\"ex"}}
第3块  {"index":0, "function":{"arguments":"pression\":"}}
第4块  {"index":0, "function":{"arguments":"\"123*456\"}"}}
```

四块拼完才是一段合法 JSON，才能 `json.loads`。而且模型可能一次要调**两个**工具，
碎块会交错到达 —— 所以必须用 `index` 当桶编号分别累积（`_merge_delta`）。

这就是「手写过 streaming agent」和「只调过 SDK」的分水岭，面试常问。

另外三个坑，代码里都处理了：

| 坑 | 后果 | 处理 |
|---|---|---|
| 开了 `include_usage` 后最后一个 chunk 没有 `choices` | `choices[0]` IndexError | 先判空再取 usage |
| 把 `reasoning_content` 塞回 messages | 多数接口直接 400 | 不回传思考过程 |
| 请求没有超时 | 网络卡住时浏览器永远转圈 | `client` 设 `timeout=60` |

### 为什么用 SSE 而不是 WebSocket

| 方案 | 特点 |
|---|---|
| 普通请求 | 必须等全部生成完，用户盯着白屏 |
| **SSE** | 单向推送、基于普通 HTTP、浏览器原生 `EventSource`、自动重连 —— 正好够用 |
| WebSocket | 双向实时，适合协同编辑/多人游戏，这里杀鸡用牛刀 |

`web_app.py` 里的生成器是同步的（`def` 不是 `async def`），
FastAPI 会自动丢进线程池 —— 所以限流退避的 `time.sleep()` 不会卡住整个服务器。
这点很重要，否则一个人被限流，所有人一起卡。

### 怎么玩

```bash
uv run web_app.py      # 打开 http://127.0.0.1:8000
```

界面上能看到：思考过程（可折叠）、每一步调了什么工具、返回了什么、
**首字延迟**、以及跑完后的步数/tokens/耗时/成本。

想体会流式的价值，问一句需要多步的问题，比如
「项目里有哪些 Python 文件？读一下 main.py 说说是干什么的」——
你会看到工具卡片先刷出来，文字再一个字一个字跟上，
而不是全部憋到最后一次性出现。

### 已验证（2026-09-08）

- SSE 链路通：`step` → `ttft_ready` → `retry` 事件都能实时推到浏览器
- 限流重试也不再是黑盒：重试提示会作为事件实时显示在页面上
- tool_call 碎片拼接：离线测试通过（两个并行工具的碎块交错到达，能正确还原）
- 内容增量渲染受限于免费档 429，高峰期（晚 18-23 点）未能跑通完整一轮

## Level 3 实测记录

| 问题 | 步数 | tokens | 说明 |
|---|---|---|---|
| 列出 Python 文件并读 level2_tools.py | 6 步 | 8161 | 真实两步链：list_files → read_file |
| 123×456 再除以 7 | 2 步 | 1566 | 模型聪明地压缩成一次 `calculator` 调用 |

## 已验证的事实

- `glm-4.7-flash` 支持 function calling，实测 0.8s 返回 tool_calls
- **它是推理模型**：响应里 `reasoning_content`（思考）和 `content`（答案）分开；
  `max_tokens` 太小时思考会吃光配额导致答案为空
- 免费档限制「同时 1 个请求」，连续两次调用常撞 429，必须重试 + 退避
- 换模型只需改 `MODEL` 一个常量；换厂商改 `base_url` 和 key 变量名
- 不给工具算 `4837*9271` 要 2020 tokens 且失败；给工具只要 217 tokens 且精确
- **DeepSeek V4 默认开启 thinking**（返回 `reasoning_content`），和智谱一样要留意
  思考 token 会算进输出计费；V4 系列原生支持 Tool Calls，可以放心当 agent 用
