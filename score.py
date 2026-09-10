"""
Level 6（第 2 块）：评分器 + 报告

这一块是 Level 6 里唯一真正有难度的地方：**怎么判定 agent 做对了没有**。

判分方式大致四种，各有各的坑：

| 方式           | 适用           | 坑                                   |
|----------------|----------------|--------------------------------------|
| 精确/包含匹配  | 计算、日期     | 太脆，换个说法就误判                 |
| 工具调用检查   | 多步任务       | agent 特有的手段，比看最终答案更稳   |
| 拒绝类判断     | 安全性         | 要判「它有没有顶住」，不是判答案     |
| LLM-as-judge   | 开放问答       | 多花一次调用，且 judge 本身也有偏差   |

本项目用前三种（判定规则写在 tasks.jsonl 里）。
成熟方案是几种混着用，**并且能说清每种的误判率** ——
能讲清这一点，比会调 LangChain 值钱得多。

用法：
    uv run score.py baseline              # 看单个实验
    uv run score.py baseline v2           # 两个实验对比（A/B）
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from level4_trace import pct

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evals"
REPORT_DIR = ROOT / "reports"
TASKS_FILE = ROOT / "tasks.jsonl"


# --------------------------------------------------------------------------
# 判分
# --------------------------------------------------------------------------
def check(rule: dict, answer: str, tools: list[str]) -> tuple[bool, str]:
    """返回 (是否通过, 原因)。原因很重要 —— 报告里要能看出「为什么没过」。"""
    t = rule["type"]

    if t == "contains":
        hit = rule["value"] in answer
        return hit, ("" if hit else f"答案里没有「{rule['value']}」")

    if t == "not_contains":
        hit = rule["value"] not in answer
        return hit, ("" if hit else f"答案里不该出现「{rule['value']}」")

    if t == "contains_any":
        hit = any(v in answer for v in rule["values"])
        return hit, ("" if hit else f"答案里没有 {rule['values']} 中任何一个")

    if t == "tool_called":
        hit = rule["name"] in tools
        return hit, ("" if hit else f"没有调用 {rule['name']}")

    if t == "tool_count_min":
        n = tools.count(rule["name"])
        hit = n >= rule["count"]
        return hit, ("" if hit else f"{rule['name']} 只调了 {n} 次，要求 ≥{rule['count']}")

    if t == "tool_sequence":
        # 子序列匹配：要求的调用按顺序出现过即可，中间允许穿插别的工具
        i = 0
        for name in tools:
            if i < len(rule["names"]) and name == rule["names"][i]:
                i += 1
        hit = i == len(rule["names"])
        return hit, ("" if hit else f"调用序列里找不到 {rule['names']} 的顺序")

    if t == "all":
        for sub in rule["rules"]:
            ok, reason = check(sub, answer, tools)
            if not ok:
                return False, reason
        return True, ""

    return False, f"未知判定类型 {t}"


def judge(rec: dict, task: dict) -> tuple[bool, str]:
    # 没正常结束（超步数、报错、被中断）一律算失败 ——
    # 一个跑不完的 agent，答案再对也不能算通过
    if rec.get("status") != "finished":
        return False, f"状态 {rec.get('status')}"
    return check(task["expect"], rec.get("answer", ""), rec.get("tools", []))


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def load(experiment: str) -> list[dict]:
    p = EVAL_DIR / f"{experiment}.jsonl"
    if not p.exists():
        print(f"找不到 {p}。先跑：uv run eval_runner.py --experiment {experiment}")
        raise SystemExit(1)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_tasks() -> dict:
    return {
        t["id"]: t
        for t in (
            json.loads(l)
            for l in TASKS_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }


def aggregate(experiment: str) -> dict:
    recs = load(experiment)
    tasks = load_tasks()

    by_task: dict[str, list[dict]] = defaultdict(list)
    judged = []
    for r in recs:
        task = tasks.get(r["task_id"])
        if not task:
            continue
        ok, reason = judge(r, task)
        r["_pass"] = ok
        r["_reason"] = reason
        judged.append(r)
        by_task[r["task_id"]].append(r)

    runs = len(judged)
    passed_runs = sum(1 for r in judged if r["_pass"])

    # pass@1 = 单次就做对的比例；pass@K = 跑 K 次至少对一次的比例。
    # 两者分开看才有意义：pass@K 高但 pass@1 低 = 不稳定，靠运气。
    task_pass = {tid: any(r["_pass"] for r in rs) for tid, rs in by_task.items()}
    task_level = sum(1 for v in task_pass.values() if v)
    n_tasks = len(by_task)

    by_cat: dict[str, dict] = defaultdict(lambda: {"tasks": 0, "passed": 0, "runs": 0, "run_pass": 0})
    for tid, rs in by_task.items():
        cat = tasks[tid].get("category", "?")
        by_cat[cat]["tasks"] += 1
        by_cat[cat]["passed"] += 1 if task_pass[tid] else 0
        by_cat[cat]["runs"] += len(rs)
        by_cat[cat]["run_pass"] += sum(1 for r in rs if r["_pass"])

    lat = [r.get("llm_wait_s", 0) for r in judged]
    return {
        "experiment": experiment,
        "runs": runs,
        "n_tasks": n_tasks,
        "pass1": passed_runs / runs if runs else 0,
        "passk": task_level / n_tasks if n_tasks else 0,
        "avg_steps": sum(r.get("steps", 0) for r in judged) / runs if runs else 0,
        "avg_tokens": sum(r.get("total_tokens", 0) for r in judged) / runs if runs else 0,
        "avg_cost": sum(r.get("cost_cny", 0) for r in judged) / runs if runs else 0,
        "total_cost": sum(r.get("cost_cny", 0) for r in judged),
        "latency_p50": pct(lat, 50),
        "latency_p95": pct(lat, 95),
        "tool_calls": sum(len(r.get("tool_calls", [])) for r in judged),
        "tool_failed": sum(
            1 for r in judged for c in r.get("tool_calls", []) if c.get("ok") is False
        ),
        "by_cat": dict(by_cat),
        "by_task": {
            tid: {
                "category": tasks[tid].get("category"),
                "passk": task_pass[tid],
                "n": len(rs),
                "npass": sum(1 for r in rs if r["_pass"]),
                "avg_steps": sum(r.get("steps", 0) for r in rs) / len(rs),
                "avg_tokens": sum(r.get("total_tokens", 0) for r in rs) / len(rs),
                "avg_cost": sum(r.get("cost_cny", 0) for r in rs) / len(rs),
                "reasons": [r["_reason"] for r in rs if not r["_pass"]],
            }
            for tid, rs in by_task.items()
        },
    }


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def pct_str(x: float) -> str:
    return f"{x * 100:.0f}%"


def render(aggs: list[dict]) -> str:
    lines: list[str] = []
    names = [a["experiment"] for a in aggs]
    lines.append(f"# Agent 评测报告：{' vs '.join(names)}")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append("| 指标 | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    rows = [
        ("任务数", lambda a: str(a["n_tasks"])),
        ("总轮次", lambda a: str(a["runs"])),
        ("**pass@1**（单次正确率）", lambda a: pct_str(a["pass1"])),
        ("**pass@K**（每题至少对一次）", lambda a: pct_str(a["passk"])),
        ("平均步数", lambda a: f"{a['avg_steps']:.2f}"),
        ("平均 tokens/轮", lambda a: f"{a['avg_tokens']:.0f}"),
        ("延迟 p50", lambda a: f"{a['latency_p50']:.2f}s"),
        ("延迟 p95", lambda a: f"{a['latency_p95']:.2f}s"),
        ("工具调用次数", lambda a: str(a["tool_calls"])),
        ("工具失败率", lambda a: pct_str(a["tool_failed"] / a["tool_calls"] if a["tool_calls"] else 0)),
        ("平均成本/轮", lambda a: f"¥{a['avg_cost']:.4f}"),
        ("总成本", lambda a: f"¥{a['total_cost']:.4f}"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(a) for a in aggs) + " |")
    lines.append("")

    # 分类
    cats = sorted({c for a in aggs for c in a["by_cat"]})
    if cats:
        lines.append("## 分类表现")
        lines.append("")
        lines.append("| 类别 | " + " | ".join(f"{n} pass@K" for n in names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for c in cats:
            cells = []
            for a in aggs:
                d = a["by_cat"].get(c)
                cells.append(f"{d['passed']}/{d['tasks']}" if d else "-")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    # 逐题
    lines.append("## 逐题明细（" + names[0] + "）")
    lines.append("")
    lines.append("| 任务 | 类别 | pass@K | 通过/次数 | 平均步数 | 平均 tokens | 失败原因 |")
    lines.append("|---|---|---|---|---|---|---|")
    for tid, d in sorted(aggs[0]["by_task"].items()):
        reason = d["reasons"][0] if d["reasons"] else "-"
        lines.append(
            f"| {tid} | {d['category']} | {'通过' if d['passk'] else '失败'} | "
            f"{d['npass']}/{d['n']} | {d['avg_steps']:.1f} | {d['avg_tokens']:.0f} | {reason} |"
        )
    lines.append("")

    if len(aggs) == 2:
        a, b = aggs
        lines.append("## A/B 结论")
        lines.append("")
        d1 = a["passk"] - b["passk"]
        d2 = a["avg_steps"] - b["avg_steps"]
        d3 = a["avg_cost"] - b["avg_cost"]
        lines.append(f"- pass@K：{pct_str(a['passk'])} vs {pct_str(b['passk'])}"
                     f"（{'高' if d1 > 0 else '低'} {abs(d1) * 100:.0f} 个百分点）")
        lines.append(f"- 平均步数：{a['avg_steps']:.2f} vs {b['avg_steps']:.2f}"
                     f"（{'多' if d2 > 0 else '少'} {abs(d2):.2f} 步）")
        lines.append(f"- 平均成本：¥{a['avg_cost']:.4f} vs ¥{b['avg_cost']:.4f}"
                     f"（{'贵' if d3 > 0 else '便宜'} {abs(d3) / max(b['avg_cost'], 1e-9) * 100:.0f}%）")
        lines.append("")
        worse = [t for t, d in a["by_task"].items() if d["passk"]
                 and not b["by_task"].get(t, {}).get("passk", True)]
        better = [t for t, d in b["by_task"].items() if d["passk"]
                  and not a["by_task"].get(t, {}).get("passk", True)]
        if better:
            lines.append(f"- {names[0]} 做对而 {names[1]} 做错的：{', '.join(better)}")
        if worse:
            lines.append(f"- {names[1]} 做对而 {names[0]} 做错的：{', '.join(worse)}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法： uv run score.py <实验名> [对比实验名]")
        raise SystemExit(1)

    names = sys.argv[1:]
    aggs = [aggregate(n) for n in names]

    md = render(aggs)
    if not REPORT_DIR.exists():
        REPORT_DIR.mkdir(parents=True)
    out = REPORT_DIR / (("-".join(names)) + ".md")
    out.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
