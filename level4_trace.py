"""
Level 4：给 Agent 加上「可观测性」（trace）

Level 3 结束后你只能看到终端上打印的东西 —— 关掉就没了，
没法回答「这 10 轮里哪一步最慢」「哪个工具报错最多」「推理 token 占了多少」。

这一级把每一次 LLM 调用、每一次工具调用都结构化地记下来，写进 JSONL 文件。

这就是 LangFuse / LangSmith / OpenTelemetry 在做的事的本质。
简历上那个「Agent 评测平台」，地基就是这里这几行。

注意看：本文件**没有重写 ReAct 循环的业务逻辑**，
工具、模型、安全限制全部直接 from level3_react import ——
加观测不该改业务代码，这本身就是一个值得记住的工程原则。

运行：
    uv run level4_trace.py              # 交互模式
    uv run analyze_traces.py            # 跑完后离线分析所有 trace
"""

import json
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

# 复用 Level 3 的一切：工具说明书、工具实现、模型名、system prompt
from level3_react import (
    MAX_STEPS,
    MODEL,
    SYSTEM_PROMPT,
    TOOLS,
    client,
    execute_tool,
)

# trace 写在这里，每天一个文件
TRACE_DIR = Path(__file__).resolve().parent / "traces"
# 不用 mkdir(exist_ok=True)：某些受限环境里目录已存在时它仍会抛错，先判断更稳
if not TRACE_DIR.exists():
    TRACE_DIR.mkdir(parents=True)


# --------------------------------------------------------------------------
# 第 1 部分：成本表
#
# 元 / 百万 token。智谱免费档是 0，但把表留在这里：
# 换付费模型时只改这一处，全项目的成本核算就跟着变。
# （价格会变，以各家官网为准）
# --------------------------------------------------------------------------
# 2026-09 DeepSeek 官方价（美元 / 百万 token），这里按 1 USD ≈ 7.1 CNY 折算成人民币：
#
#   模型                        输入(缓存未命中)   输出      并发上限
#   deepseek-v4-flash           $0.22 闲/$0.44 峰  $0.66/$1.32   2500
#   deepseek-v4-pro             $0.66 闲/$1.32 峰  $1.98/$3.96    500
#
# 两个折扣，本表取的是「峰值 + 缓存未命中」这个最贵的口径（保守估算）：
#   · 时段：峰值 = UTC 01-04、06-10（北京时间 9-12 点 / 14-18 点，工作日），
#           其余时间半价。所以晚上跑是半价。
#   · 缓存命中：输入便宜约 30 倍（$0.014 vs $0.44），
#           把不变的 system prompt 放最前面能吃到这个折扣。
PRICING = {
    # 智谱免费档
    "glm-4.7-flash": (0.0, 0.0),
    "glm-5": (10.0, 30.0),
    # DeepSeek V4（人民币 / 百万 token，峰值未命中口径）
    "deepseek-v4-flash": (3.12, 9.37),
    "deepseek-v4-pro": (9.37, 28.12),
    "deepseek-v4-flash-vision-exp": (3.12, 9.37),
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out


# --------------------------------------------------------------------------
# 第 2 部分：TraceLogger
#
# JSONL = 每行一个 JSON。为什么不用一个大 JSON 数组？
#   1. 可以追加写，程序崩了也不丢前面的数据
#   2. 可以一行行流式读，百万行也不占内存
#   3. jq / pandas 都能直接吃
# --------------------------------------------------------------------------
class TraceLogger:
    def __init__(self, meta: dict | None = None) -> None:
        # meta 会写进这一轮的每一条事件里。
        # Level 6 用它打实验标签（experiment / task_id / repeat），
        # 这样几十轮跑完之后，才能从同一个 trace 文件里把不同实验分开统计。
        self.meta: dict = meta or {}
        self.run_id = uuid.uuid4().hex[:8]
        self.path = TRACE_DIR / f"{datetime.now():%Y%m%d}.jsonl"
        self.events: list[dict] = []

    def log(self, event: str, **data) -> dict:
        record = {
            "run_id": self.run_id,
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **self.meta,
            **data,
        }
        self.events.append(record)
        # 每次都打开-追加-关闭。看起来浪费，但保证中途崩溃也不丢数据。
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def summary(self) -> dict:
        """从本轮已记录的事件里算出统计值。"""
        llm = [e for e in self.events if e["event"] == "llm_call"]
        tools = [e for e in self.events if e["event"] == "tool_call"]
        errors = [e for e in self.events if e["event"] == "llm_error"]

        llm_tokens = sum(e["total_tokens"] for e in llm)
        reasoning = sum(e.get("reasoning_tokens", 0) for e in llm)
        llm_wait = sum(e["latency_s"] for e in llm)
        tool_wait = sum(e["latency_s"] for e in tools)

        return {
            "run_id": self.run_id,
            "steps": max((e.get("step", 0) for e in llm), default=0),
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_failed": sum(1 for e in tools if not e.get("ok")),
            "total_tokens": llm_tokens,
            "reasoning_tokens": reasoning,
            "llm_wait_s": round(llm_wait, 2),
            "tool_wait_s": round(tool_wait, 3),
            "retries": sum(e.get("retries", 0) for e in llm) + len(errors),
            "cost_cny": round(sum(e.get("cost_cny", 0) for e in llm), 6),
        }


# --------------------------------------------------------------------------
# 第 3 部分：带记录的请求
# --------------------------------------------------------------------------
def chat_traced(messages, logger: TraceLogger, step: int, retries: int = 6):
    """和 Level 3 的 chat 一样，但把耗时、token、重试次数都记下来。"""
    for i in range(retries):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
            )
            latency = time.perf_counter() - t0

            u = response.usage
            details = getattr(u, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) or 0

            logger.log(
                "llm_call",
                step=step,
                latency_s=round(latency, 2),
                retries=i,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                reasoning_tokens=reasoning,
                total_tokens=u.total_tokens,
                cost_cny=round(cost_of(MODEL, u.prompt_tokens, u.completion_tokens), 6),
            )
            return response

        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                logger.log("llm_error", step=step, error=type(e).__name__,
                           message=str(e)[:200])
                raise
            wait = 4 * (i + 1)
            print(f"      ({type(e).__name__}，等待 {wait}s 后重试 {i + 1}/{retries - 1}，Ctrl+C 可放弃)")
            time.sleep(wait)


# --------------------------------------------------------------------------
# 第 4 部分：主循环（逻辑和 Level 3 完全一致，只是多了 logger.log）
# --------------------------------------------------------------------------
# 工具返回这些前缀，说明这次调用没成功
# （Level 5 会 import 它，所以不加下划线前缀）
FAILURE_PREFIXES = (
    "错误", "工具执行出错", "安全限制", "文件不存在", "目录不存在", "计算失败", "不是文件", "不是目录",
)


def run(question: str, logger: TraceLogger) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    logger.log("run_start", question=question, model=MODEL, max_steps=MAX_STEPS)

    for step in range(1, MAX_STEPS + 1):
        print(f"\n---- 第 {step} 步 ----")
        response = chat_traced(messages, logger, step)
        message = response.choices[0].message

        if message.content:
            print(f"  想法: {message.content.strip()[:300]}")

        if not message.tool_calls:
            logger.log("run_end", step=step, status="finished",
                       answer=message.content)
            print(f"\n===== 最终回答 =====\n{message.content}")
            return

        messages.append(message)
        print(f"  行动: 请求调用 {len(message.tool_calls)} 个工具")

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)

            t0 = time.perf_counter()
            result = execute_tool(tc.function.name, args)
            tool_latency = time.perf_counter() - t0

            ok = not result.startswith(FAILURE_PREFIXES)
            logger.log(
                "tool_call",
                step=step,
                tool=tc.function.name,
                args=args,
                ok=ok,
                latency_s=round(tool_latency, 3),
                result_chars=len(result),
                result_preview=result[:120],
            )

            preview = result if len(result) <= 200 else result[:200] + " ...(已截断)"
            print(f"    > {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            print(f"      返回: {preview}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    logger.log("run_end", status="max_steps_exceeded")
    print(f"\n!! 达到最大步数 {MAX_STEPS}，强制结束")


def pct(values: list[float], p: float) -> float:
    """
    最近秩法（nearest-rank）算分位数。

    为什么不用 statistics.quantiles：它默认做插值，
    会算出一个比样本最大值还大的 p95（实测出现过 22.5s > 实测最大 21.1s），
    看报表的人会以为数据错了。最近秩法保证结果一定是样本里的某个真实值。
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, -(-int(p * len(s)) // 100) - 1))
    return s[k]


def print_summary(logger: TraceLogger) -> None:
    s = logger.summary()
    llm = [e for e in logger.events if e["event"] == "llm_call"]
    lat = [e["latency_s"] for e in llm]

    p95 = round(pct(lat, 95), 2)
    ratio = (s["reasoning_tokens"] / s["total_tokens"] * 100) if s["total_tokens"] else 0

    print("\n" + "=" * 52)
    print("本轮 trace 统计")
    print("=" * 52)
    print(f"  步数            {s['steps']}")
    print(f"  LLM 调用        {s['llm_calls']} 次（重试 {s['retries']} 次）")
    print(f"  工具调用        {s['tool_calls']} 次（失败 {s['tool_failed']} 次）")
    print(f"  tokens          {s['total_tokens']}（其中推理 {s['reasoning_tokens']}，占 {ratio:.0f}%）")
    print(f"  等待 LLM        {s['llm_wait_s']}s（单次 p95 {p95}s）")
    print(f"  执行工具        {s['tool_wait_s']}s")
    print(f"  预估成本        ¥{s['cost_cny']}")
    print(f"  已写入          {logger.path}")
    print("=" * 52)


def main() -> None:
    print("=" * 56)
    print("Level 4：带 trace 的 ReAct 循环（输入 q 回车退出）")
    print("=" * 56)
    print("跑几轮之后执行：uv run analyze_traces.py")

    while True:
        try:
            q = input("\n你（输入 q 回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        q = unicodedata.normalize("NFKC", q).strip()
        if q.lower() in ("q", "quit", "exit", "退出"):
            break
        if not q:
            continue

        logger = TraceLogger()
        try:
            run(q, logger)
            print_summary(logger)
        except KeyboardInterrupt:
            logger.log("run_end", status="interrupted")
            print("\n（已打断本轮）")
        except Exception as e:  # noqa: BLE001
            logger.log("run_end", status="error", error=type(e).__name__)
            msg = str(e)
            if "429" in msg or "1305" in msg:
                print("\n限流了：免费档高峰期常见，等一两分钟再试。")
                print("（这次的失败也记进 trace 了 —— 失败数据同样有价值）")
            else:
                print(f"\n出错了：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
