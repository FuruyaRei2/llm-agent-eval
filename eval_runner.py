"""
Level 6（第 1 块）：批量执行器

Level 4 解决了「记录」，Level 5 解决了「看得见」，
但到 Level 5 为止，你依然回答不了一个问题：**我的 agent 到底有多好？**

要回答它，得做三件事：
    1. 有一批固定的题（tasks.jsonl）
    2. 每题跑 K 遍（因为 agent 输出是随机的，跑一次说明不了任何问题）
    3. 判分 + 聚合（score.py）

这个文件干第 2 件。它复用 Level 5 的 run_stream —— 评测不应该另写一套
agent 逻辑，否则你测的就不是线上那个 agent 了，这是个很容易犯的错。

用法：
    uv run eval_runner.py --experiment baseline              # 全量跑，每题 3 遍
    uv run eval_runner.py --experiment baseline --limit 3    # 先跑 3 题试试水
    uv run eval_runner.py --experiment baseline --repeat 5   # 每题 5 遍
    uv run eval_runner.py --experiment baseline --category safety   # 只跑安全类
    uv run eval_runner.py --experiment baseline --dry-run    # 只估成本，不真跑

跑完的结果写在 evals/<experiment>.jsonl，然后：
    uv run score.py baseline
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from level4_trace import TraceLogger
from level5_stream import run_stream

ROOT = Path(__file__).resolve().parent
TASKS_FILE = ROOT / "tasks.jsonl"
EVAL_DIR = ROOT / "evals"

_print_lock = threading.Lock()


def load_tasks(category: str | None = None, limit: int | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in TASKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if category:
        tasks = [t for t in tasks if t.get("category") == category]
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_one(task: dict, experiment: str, repeat: int) -> dict:
    """
    跑一个任务的一次。返回一条「结果记录」。

    注意这里 TraceLogger 传了 meta：experiment / task_id / repeat 会写进
    这一轮的每一条事件。几十轮跑完后，trace 文件里所有实验混在一起，
    靠这几个标签才能把它们分开——这就是为什么 meta 要在 Logger 层做，
    而不是事后在结果文件里补。
    """
    logger = TraceLogger(
        meta={
            "experiment": experiment,
            "task_id": task["id"],
            "category": task.get("category"),
            "repeat": repeat,
        }
    )

    answer_parts: list[str] = []
    calls: list[dict] = []
    status = "unknown"
    stats: dict = {}
    error = None

    try:
        for ev in run_stream(task["question"], logger):
            t = ev["type"]
            if t == "answer_delta":
                answer_parts.append(ev["text"])
            elif t == "tool_call":
                calls.append({"name": ev["name"], "args": ev["args"], "ok": None})
            elif t == "tool_result":
                if calls:
                    calls[-1]["ok"] = ev["ok"]
                    calls[-1]["latency_s"] = ev["latency_s"]
            elif t == "done":
                status = ev["status"]
                stats = ev.get("stats", {})
    except Exception as e:  # noqa: BLE001
        # 单次失败不能让整个评测崩掉 —— 失败本身也是要统计的数据
        status = "error"
        error = f"{type(e).__name__}: {e}"

    return {
        "experiment": experiment,
        "task_id": task["id"],
        "category": task.get("category"),
        "repeat": repeat,
        "run_id": logger.run_id,
        "status": status,
        "answer": "".join(answer_parts),
        "tool_calls": calls,
        "tools": [c["name"] for c in calls],
        "steps": stats.get("steps", 0),
        "llm_calls": stats.get("llm_calls", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "reasoning_tokens": stats.get("reasoning_tokens", 0),
        "llm_wait_s": stats.get("llm_wait_s", 0.0),
        "tool_wait_s": stats.get("tool_wait_s", 0.0),
        "retries": stats.get("retries", 0),
        "cost_cny": stats.get("cost_cny", 0.0),
        "error": error,
    }


def estimate(tasks: list[dict], repeat: int) -> None:
    """开跑前先给个成本估算。花别人的钱（哪怕是自己的）之前要有数。"""
    runs = len(tasks) * repeat
    print("=" * 56)
    print("成本估算（按实测经验值粗估）")
    print("=" * 56)
    print(f"  任务数        {len(tasks)}")
    print(f"  每个任务跑    {repeat} 次")
    print(f"  总调用轮次    {runs}")
    print(f"  粗估 token    {runs * 3000:,} 左右（多步任务会更贵）")
    print("  粗估花费      几毛到 2 元之间，取决于步数和思考 token")
    print("  建议          先用 --limit 3 跑通流程，再全量跑")
    print("=" * 56)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="实验名，用来区分不同配置的结果")
    ap.add_argument("--repeat", type=int, default=3, help="每个任务跑几次，默认 3")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个任务")
    ap.add_argument("--category", default=None, help="只跑某一类任务")
    ap.add_argument("--concurrency", type=int, default=4, help="并发数，默认 4")
    ap.add_argument("--dry-run", action="store_true", help="只估算，不真跑")
    args = ap.parse_args()

    tasks = load_tasks(args.category, args.limit)
    if not tasks:
        print("没匹配到任务。检查 --category 或 --limit。")
        raise SystemExit(1)

    estimate(tasks, args.repeat)
    if args.dry_run:
        return

    # 目录在真正要写的时候才建。别放在 import 时做副作用 ——
    # 那样「只是导入一下这个模块」也会去动文件系统，既慢又容易在受限环境里卡住
    if not EVAL_DIR.exists():
        EVAL_DIR.mkdir(parents=True)
    out_path = EVAL_DIR / f"{args.experiment}.jsonl"
    print(f"\n开始跑：实验 {args.experiment} | 并发 {args.concurrency}")
    print(f"结果写入 {out_path}\n")

    jobs = [(t, r) for t in tasks for r in range(args.repeat)]
    done = 0
    results: list[dict] = []

    # 并发跑。DeepSeek 并发上限 2500，这里保守用 4 ——
    # 并发太高会掩盖限流问题，也不利于复现
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, t, args.experiment, r): (t, r) for t, r in jobs}
        for fu in as_completed(futures):
            t, r = futures[fu]
            try:
                rec = fu.result()
            except Exception as e:  # noqa: BLE001
                rec = {"task_id": t["id"], "status": "error", "error": repr(e)}
            results.append(rec)
            done += 1
            with _print_lock:
                mark = "OK " if rec.get("status") == "finished" else "!! "
                print(f"  [{done}/{len(jobs)}] {mark}{rec.get('task_id')}"
                      f" 第{r + 1}次  状态={rec.get('status')}"
                      f"  步数={rec.get('steps', 0)}"
                      f"  tokens={rec.get('total_tokens', 0)}")

    # 一次性写入（结果文件按实验分开，重跑会覆盖上一次的）
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r.get("status") == "finished")
    cost = sum(r.get("cost_cny", 0) for r in results)
    tokens = sum(r.get("total_tokens", 0) for r in results)

    print()
    print("=" * 56)
    print("跑完了")
    print("=" * 56)
    print(f"  总轮次    {len(results)}（正常结束 {ok}，异常 {len(results) - ok}）")
    print(f"  总 tokens {tokens:,}")
    print(f"  实际花费  ¥{cost:.4f}")
    print(f"  结果文件  {out_path}")
    print("\n下一步： uv run score.py " + args.experiment)


if __name__ == "__main__":
    main()
