"""
trace 分析器（纯离线，不花一分钱、不联网）

跑完几轮 level4 之后执行：

    uv run analyze_traces.py

它会回答这些 Level 3 时代根本答不上来的问题：
    · 一共跑了几轮、烧了多少 token
    · 每一步的延迟分布（p50 / p95 / 最慢）
    · 推理 token 占总消耗多少 —— 这个数字通常高得吓人
    · 哪个工具被调用最多、哪个最容易失败
    · 有多少次被限流重试

这个文件里的统计逻辑，和你简历项目里要做的「失败模式分析」是同一套东西。
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parent / "traces"


def load_events(paths=None):
    """默认读 traces/ 下所有 .jsonl；也可以 uv run analyze_traces.py <文件> 指定。"""
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = sorted(TRACE_DIR.glob("*.jsonl"))
    if not files:
        return [], []
    events = []
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events, files


def pct(values, p):
    """
    取百分位数。

    不用 statistics.quantiles —— 它会做线性外推，
    样本少的时候能算出比最大值还大的 p95（比如只有 4 个样本时 p95 > max），
    那是个会误导人的假数字。这里用最近秩法，结果一定落在真实样本范围内。
    """
    if not values:
        return 0
    s = sorted(values)
    idx = round((p / 100) * (len(s) - 1))
    idx = max(0, min(len(s) - 1, idx))  # 夹在合法范围内
    return round(s[idx], 2)


def bar(n, total, width=20):
    """画一个简单的横向条，纯文本。"""
    if total <= 0:
        return ""
    filled = max(1, round(n / total * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    events, files = load_events(sys.argv[1:] or None)
    if not events:
        print("还没有任何 trace。先去跑几轮：uv run level4_trace.py")
        return

    runs: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        runs[e["run_id"]].append(e)

    llm = [e for e in events if e["event"] == "llm_call"]
    tools = [e for e in events if e["event"] == "tool_call"]
    errors = [e for e in events if e["event"] == "llm_error"]

    print("=" * 60)
    print(f"trace 分析  |  {len(files)} 个文件  |  {len(runs)} 轮对话")
    print("=" * 60)

    # ---------- 总览 ----------
    total_tokens = sum(e["total_tokens"] for e in llm)
    reasoning = sum(e.get("reasoning_tokens", 0) for e in llm)
    prompt = sum(e["prompt_tokens"] for e in llm)
    cost = sum(e.get("cost_cny", 0) for e in llm)

    print("\n[总览]")
    print(f"  LLM 调用      {len(llm)} 次")
    print(f"  工具调用      {len(tools)} 次")
    print(f"  限流/错误     {len(errors)} 次")
    print(f"  总 tokens     {total_tokens}")
    print(f"    输入        {prompt}")
    print(f"    输出        {total_tokens - prompt}")
    print(f"    其中推理    {reasoning}  ({reasoning / total_tokens * 100:.0f}%)" if total_tokens else "")
    print(f"  预估成本      ¥{cost:.4f}")

    # ---------- 延迟 ----------
    lat = [e["latency_s"] for e in llm]
    if lat:
        print("\n[每次 LLM 调用的延迟]")
        print(f"  最快   {min(lat):.2f}s")
        print(f"  p50    {pct(lat, 50):.2f}s")
        print(f"  p95    {pct(lat, 95):.2f}s")
        print(f"  最慢   {max(lat):.2f}s")
        print(f"  平均   {statistics.mean(lat):.2f}s")
        retry_total = sum(e.get("retries", 0) for e in llm)
        if retry_total:
            print(f"  重试   {retry_total} 次（免费档限流，会显著拉长体感延迟）")

    # ---------- 工具 ----------
    if tools:
        print("\n[工具调用]")
        by_tool = Counter(e["tool"] for e in tools)
        for name, cnt in by_tool.most_common():
            sub = [e for e in tools if e["tool"] == name]
            ok = sum(1 for e in sub if e.get("ok"))
            avg = statistics.mean(e["latency_s"] for e in sub)
            print(f"  {name:16s} {cnt:3d} 次  成功 {ok}/{cnt}  平均 {avg * 1000:.1f}ms")
            print(f"    {bar(cnt, len(tools))}")

        failed = [e for e in tools if not e.get("ok")]
        if failed:
            print("\n  失败明细：")
            for e in failed[:8]:
                p = e.get("result_preview", "")[:70]
                print(f"    {e['tool']}({json.dumps(e.get('args', {}), ensure_ascii=False)})")
                print(f"      -> {p}")

    # ---------- 每轮一览 ----------
    print("\n[每轮概览]")
    print(f"  {'run_id':<10}{'步数':<6}{'LLM':<6}{'工具':<6}{'tokens':<10}{'耗时':<10}结局")
    print("  " + "-" * 56)
    for run_id, evs in runs.items():
        sub_llm = [e for e in evs if e["event"] == "llm_call"]
        sub_tool = [e for e in evs if e["event"] == "tool_call"]
        end = next((e for e in evs if e["event"] == "run_end"), None)
        status = end.get("status", "?") if end else "中断/未结束"
        tk = sum(e["total_tokens"] for e in sub_llm)
        secs = sum(e["latency_s"] for e in sub_llm)
        steps = max((e.get("step", 0) for e in sub_llm), default=0)
        print(f"  {run_id:<10}{steps:<6}{len(sub_llm):<6}{len(sub_tool):<6}{tk:<10}{secs:<10.1f}{status}")

    print("\n提示：trace 是追加写的，越攒越多。想清空就删掉 traces/ 目录。")
    print("下一步可以做的：加一个 --last N 只看最近 N 轮，或者按 tool 分组画耗时图。")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
