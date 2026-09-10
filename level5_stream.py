"""
Level 5：流式输出（streaming）

Level 4 之前，每问一句都要等模型把整段话全部生成完，才一次性打印出来。
真实产品不会这样 —— ChatGPT 是一个字一个字往外蹦的。这一级就来做这件事。

======================================================================
这一级真正的难点（也是面试常被问的点）
======================================================================
流式 + 工具调用 放在一起时，会撞上一个很别扭的问题：

    普通文本是一块块来的，直接拼就行：
        "你" "好" "，" "我" "是"  ->  "你好，我是"

    但工具调用是「结构化」的，它也被切碎了发过来：
        第1块  {"index":0, "id":"call_a", "function":{"name":"calc","arguments":""}}
        第2块  {"index":0, "function":{"arguments":"{\"ex"}}
        第3块  {"index":0, "function":{"arguments":"pression\":"}}
        第4块  {"index":0, "function":{"arguments":"\"123*456\"}"}}

    同一个工具的参数被拆成好几段，你必须自己按 index 把它们拼回去，
    拼完之后 arguments 才是一段合法的 JSON，才能 json.loads。

    更麻烦的是模型可能一次要调好几个工具（并行调用），
    碎块是交错到达的 —— 所以必须用 index 当桶的编号分别累积，
    不能简单地追加到一个字符串里。

这段代码里 `_merge_delta()` 就是干这个的，只有十几行，但它是
「手写 streaming agent」和「只会调 SDK」的分水岭。

另外两个容易踩的点，代码里都处理了：
    1. 开启 include_usage 后，最后一个 chunk 没有 choices，只有 usage
       —— 不判断会 IndexError
    2. 思考过程（reasoning_content）不能塞回 messages
       —— 回传会让很多接口直接 400

运行：
    uv run level5_stream.py          # 终端里看流式
    uv run web_app.py                # 网页界面（Level 5 第二部分）
"""

import json
import time
import unicodedata

from level3_react import (
    MAX_STEPS,
    MODEL,
    SYSTEM_PROMPT,
    TOOLS,
    client,
    execute_tool,
)
from level4_trace import (
    FAILURE_PREFIXES,
    TraceLogger,
    cost_of,
    print_summary,
)


# --------------------------------------------------------------------------
# 第 1 部分：把切碎的 tool_call 拼回去
# --------------------------------------------------------------------------
def _extra(obj, key):
    """
    取一个可能不在 SDK 类型定义里的字段。

    不同厂商会在响应里塞私有字段（比如智谱的 reasoning_content）。
    pydantic 有时把它挂成属性，有时塞进 model_extra，两边都得试。
    """
    v = getattr(obj, key, None)
    if v is None:
        v = (getattr(obj, "model_extra", None) or {}).get(key)
    return v


def _merge_delta(acc: dict, delta_tc) -> None:
    """
    把一个 tool_call 碎块合并进累加器。

    acc 长这样：{0: {"id": "", "name": "", "arguments": ""}, 1: {...}}
    键是 index —— 模型要求并行调用两个工具时，index 分别是 0 和 1，
    碎块交错到达，靠这个编号才知道该往哪个桶里追加。
    """
    idx = delta_tc.index if delta_tc.index is not None else 0
    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})

    if delta_tc.id:
        slot["id"] += delta_tc.id

    fn = getattr(delta_tc, "function", None)
    if fn is not None:
        if fn.name:
            slot["name"] += fn.name
        if fn.arguments:
            slot["arguments"] += fn.arguments


# --------------------------------------------------------------------------
# 第 2 部分：一次流式 LLM 调用
# --------------------------------------------------------------------------
def stream_step(messages, logger: TraceLogger, step: int, retries: int = 6):
    """
    发一次流式请求。它是一个生成器（generator）——

        for kind, payload in stream_step(...):
            ...

    边收边 yield，所以调用方能「实时」拿到内容，而不是等全部结束。

    kind 有四种：
        "reasoning"  思考过程的一小段（推理模型才有）
        "content"    正文的一小段
        "retry"      被限流了，正在等待重试
        "done"       流结束，payload 是拼好的 assistant 消息（可直接 append 进 messages）
    """
    attempt = 0
    include_usage = True  # 先按标准做法要用量，不支持的厂商会自动降级

    while True:
        # 每次重试都要把缓冲区清空 —— 上一次可能已经流了一半
        acc: dict = {}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        t0 = time.perf_counter()

        try:
            kwargs = {
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "stream": True,
            }
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}

            stream = client.chat.completions.create(**kwargs)

            for chunk in stream:
                # 用量可能在任何 chunk 里出现，两种风格都要接住：
                #   OpenAI 官方：最后一个 chunk 只有 usage，没有 choices
                #   DeepSeek：  最后一个 chunk choices 和 usage 一起给
                # 只在「没有 choices」的分支里读 usage 会漏掉后者 —— 实测踩过。
                if getattr(chunk, "usage", None):
                    usage = chunk.usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                r = _extra(delta, "reasoning_content")
                if r:
                    reasoning_parts.append(r)
                    yield ("reasoning", r)

                if delta.content:
                    content_parts.append(delta.content)
                    yield ("content", delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        _merge_delta(acc, tc)

        except Exception as e:  # noqa: BLE001
            msg = str(e)

            # 有些接口不认 stream_options，报 400。降级后重试，只是拿不到 token 统计。
            if include_usage and ("stream_options" in msg or "include_usage" in msg):
                include_usage = False
                print("      (该接口不支持 stream_options，降级为不含用量统计的流式)")
                continue

            attempt += 1
            if attempt >= retries:
                logger.log(
                    "llm_error", step=step, error=type(e).__name__, message=msg[:200]
                )
                raise

            wait = 4 * attempt
            yield ("retry", f"{type(e).__name__}，等待 {wait}s 后重试 {attempt}/{retries - 1}")
            time.sleep(wait)
            continue

        break  # 整条流正常走完

    # ---- 流结束了，把碎片拼成完整结果 ----
    latency = time.perf_counter() - t0
    content = "".join(content_parts)

    tool_calls = []
    for idx in sorted(acc):
        slot = acc[idx]
        try:
            args = json.loads(slot["arguments"]) if slot["arguments"].strip() else {}
        except json.JSONDecodeError:
            # 参数没拼全（连接中断、模型输出不合法）时别让整个 agent 崩掉
            args = {}
        tool_calls.append(
            {
                "id": slot["id"] or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )

    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        # 缓存命中的输入 token：DeepSeek 对这部分收 1/30 的钱。
        # 字段名各家不一：DeepSeek 在顶层，OpenAI 风格在 prompt_tokens_details 里
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if cache_hit is None:
            cache_hit = getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
            ) or 0
        logger.log(
            "llm_call",
            step=step,
            latency_s=round(latency, 2),
            retries=attempt,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_hit_tokens=cache_hit,
            total_tokens=usage.total_tokens,
            cost_cny=round(cost_of(MODEL, usage.prompt_tokens, usage.completion_tokens), 6),
            streamed=True,
        )
    else:
        # 接口不给用量，也照样记一条 —— 延迟和重试次数才是最有价值的数据
        logger.log(
            "llm_call",
            step=step,
            latency_s=round(latency, 2),
            retries=attempt,
            prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
            cost_cny=0.0,
            streamed=True,
            usage_unavailable=True,
        )

    assistant_msg = {"role": "assistant", "content": content or None}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    # 故意不写 reasoning_content：思考过程是模型的内部状态，
    # 回传进 messages 会让 OpenAI 及多数兼容接口直接 400。

    yield ("done", assistant_msg)


# --------------------------------------------------------------------------
# 第 3 部分：流式的 ReAct 主循环
#
# 逻辑和 Level 4 一模一样，唯一区别是它不 return 结果，
# 而是把每一步都 yield 成事件。终端和网页共用这一个生成器 ——
# 换界面不该重写业务逻辑，这和 Level 4 那条原则是一回事。
# --------------------------------------------------------------------------
def run_stream(question: str, logger: TraceLogger):
    """边跑边产出事件。事件是普通 dict，方便直接 JSON 序列化推给浏览器。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    logger.log("run_start", question=question, model=MODEL, max_steps=MAX_STEPS,
               mode="stream")

    answer_parts: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        yield {"type": "step", "step": step}
        yield {"type": "ttft_ready", "step": step}  # 前端用它算「首字延迟」

        msg = None
        for kind, payload in stream_step(messages, logger, step):
            if kind == "reasoning":
                yield {"type": "reasoning", "text": payload}
            elif kind == "content":
                answer_parts.append(payload)
                yield {"type": "answer_delta", "text": payload}
            elif kind == "retry":
                yield {"type": "retry", "text": payload}
            elif kind == "done":
                msg = payload

        tool_calls = (msg or {}).get("tool_calls") or []

        # 终止条件 1：模型没要工具 —— 说明它认为信息够了
        if not tool_calls:
            answer = "".join(answer_parts)
            # 把最终答案记进 trace：评测时要靠它判分，
            # 光有 token 和耗时是判断不了「做对没有」的
            logger.log("run_end", step=step, status="finished", answer=answer)
            yield {"type": "done", "status": "finished", "stats": logger.summary()}
            return

        messages.append(msg)

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            yield {"type": "tool_call", "name": name, "args": args}

            t0 = time.perf_counter()
            result = execute_tool(name, args)
            tool_latency = time.perf_counter() - t0

            ok = not result.startswith(FAILURE_PREFIXES)
            logger.log(
                "tool_call",
                step=step,
                tool=name,
                args=args,
                ok=ok,
                latency_s=round(tool_latency, 3),
                result_chars=len(result),
                result_preview=result[:120],
            )
            yield {
                "type": "tool_result",
                "name": name,
                "result": result,
                "ok": ok,
                "latency_s": round(tool_latency, 3),
            }

            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # 终止条件 2：安全阀
    logger.log("run_end", status="max_steps_exceeded")
    yield {"type": "done", "status": "max_steps_exceeded", "stats": logger.summary()}


# --------------------------------------------------------------------------
# 第 4 部分：终端界面
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 56)
    print("Level 5：流式 ReAct（输入 q 回车退出，随时 Ctrl+C）")
    print("=" * 56)
    print("重点观察：文字是一个字一个字蹦出来的，但工具调用必须")
    print("等参数全部收齐才能执行 —— 这就是流式的核心矛盾。")
    print("想看网页版：uv run web_app.py")

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
        prefix = ""  # 每段流前面加什么标签（"  思考: " / "  回答: "）

        try:
            for ev in run_stream(q, logger):
                t = ev["type"]

                if t == "step":
                    print(f"\n---- 第 {ev['step']} 步 ----", end="", flush=True)
                    prefix = ""
                elif t == "reasoning":
                    if prefix != "  思考: ":
                        print("\n  思考: ", end="", flush=True)
                        prefix = "  思考: "
                    print(ev["text"], end="", flush=True)
                elif t == "answer_delta":
                    if prefix != "\n  回答: ":
                        print("\n  回答: ", end="", flush=True)
                        prefix = "\n  回答: "
                    print(ev["text"], end="", flush=True)
                elif t == "retry":
                    print(f"\n      ({ev['text']}, Ctrl+C 可放弃)", flush=True)
                    prefix = ""
                elif t == "tool_call":
                    args = json.dumps(ev["args"], ensure_ascii=False)
                    print(f"\n  > 调用 {ev['name']}({args})", flush=True)
                    prefix = ""
                elif t == "tool_result":
                    r = ev["result"]
                    preview = r if len(r) <= 200 else r[:200] + " ...(已截断)"
                    print(f"    返回: {preview}", flush=True)
                    prefix = ""
                elif t == "done":
                    print()
                    if ev["status"] == "max_steps_exceeded":
                        print(f"\n!! 达到最大步数 {MAX_STEPS}，强制结束")

            print_summary(logger)

        except KeyboardInterrupt:
            logger.log("run_end", status="interrupted")
            print("\n（已打断本轮）")
        except Exception as e:  # noqa: BLE001
            logger.log("run_end", status="error", error=type(e).__name__)
            msg = str(e)
            if "429" in msg or "1305" in msg:
                print("\n限流了：免费档同时只能 1 个请求。等一两分钟再试。")
            else:
                print(f"\n出错了：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
