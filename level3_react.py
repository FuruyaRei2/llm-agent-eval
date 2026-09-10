"""
Level 3：完整的 ReAct 循环

Level 2 只能调「一轮」工具 —— 拿到结果就直接组织答案了。
但真实问题往往需要连着调好几次：先列目录才知道要读哪个文件，
先算乘法才能做除法，先查时间才能判断该不该提醒。

这一级把「请求 -> 执行 -> 回灌」包进一个 while 循环，让模型自己决定：
    · 还要不要再调一次工具？
    · 还是信息够了，可以回答了？

循环结束的条件只有两个：
    1. 模型不再请求工具（它认为够了）
    2. 达到最大步数上限（防止它无限打转 —— agent 必备的安全阀）

ReAct = Reasoning + Acting，就是这个循环的名字。

运行：
    uv run level3_react.py
试试问：
    这个项目里有哪些 Python 文件？读一下 main.py 告诉我它是干什么的
    先算 123 乘以 456，再把结果除以 7
    看看项目里最大的文件是哪个，然后算出它的大小是几 KB
"""

import ast
import json
import operator
import os
import random
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

# ==========================================================================
# 厂商配置：换模型只改 .env，不动代码
#
# 三级优先级：
#   1. 显式三件套 LLM_API_KEY + LLM_BASE_URL + MODEL（接任何 OpenAI 兼容接口）
#   2. 厂商专用变量：按顺序找，先填中的先用
#   3. 都没有 -> 报错退出
#
# 想换模型（比如 flash 换成 pro）不用动这里，在 .env 里加一行 MODEL=xxx 即可。
# ==========================================================================
PROVIDERS = [
    # (环境变量, base_url, 默认模型)
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash"),
    ("ZAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4/", "glm-4.7-flash"),
]

def looks_like_key(v: str) -> bool:
    """
    判断一个字符串是不是真 key（而不是没换掉的占位符）。

    两个条件：长度够 + 不含中文。
    后者很关键 —— 占位符「在这里粘贴你的密钥」有 15 个字符，
    光看长度容易漏，但密钥里绝不可能出现中文。
    """
    if len(v) < 20:
        return False
    return not any("\u4e00" <= ch <= "\u9fff" for ch in v)


api_key = os.environ.get("LLM_API_KEY", "").strip()
BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
MODEL = os.environ.get("MODEL", "").strip()
PROVIDER = "自定义"

if not looks_like_key(api_key):
    api_key = ""
    for _var, _url, _default_model in PROVIDERS:
        _v = os.environ.get(_var, "").strip()
        if looks_like_key(_v):
            api_key = _v
            PROVIDER = _var.replace("_API_KEY", "")
            BASE_URL = BASE_URL or _url
            MODEL = MODEL or _default_model
            break

if not api_key:
    print("还没填 API Key。打开 .env，填 DEEPSEEK_API_KEY（付费）或 ZAI_API_KEY（免费）。")
    print("注意：占位符里的中文要整段替换掉，不能只改一半。")
    raise SystemExit(1)

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    # 必须设超时。默认没超时的话，一旦网络卡住（实测免费档限流时遇到过），
    # 请求会永久挂着 —— 流式输出时表现为浏览器一直转圈但永远不报错。
    timeout=60.0,
)

# 循环最多转几步。这是 agent 最重要的安全阀之一：
# 没有它，模型一旦陷入「调工具 -> 结果不满意 -> 再调一次」的循环，
# 你的钱和token会在几秒内烧完。
MAX_STEPS = 8

# ==========================================================================
# 第 1 部分：工具的「说明书」
# ==========================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。需要精确数值结果时使用。一次只能算一个表达式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '123 * 456' 或 '(88 + 12) / 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。问现在几点、今天几号、星期几时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "掷骰子，返回随机整数。模型自己无法产生真随机，必须用它。",
            "parameters": {
                "type": "object",
                "properties": {"sides": {"type": "integer", "description": "骰子面数，默认 6"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出某个目录下的文件和子目录。想知道有哪些文件时先调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "相对项目根目录的路径，'.' 表示根目录，默认 '.'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件的内容。先用 list_files 确认文件存在再调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对项目根目录的文件路径"},
                    "max_chars": {
                        "type": "integer",
                        "description": "最多读取多少字符，默认 2000",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


# ==========================================================================
# 第 2 部分：工具的「真实实现」
# ==========================================================================

# --- 计算器：绝不能用 eval()，那是把电脑控制权交给别人 ---
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


def calculator(expression: str) -> str:
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def roll_dice(sides: int = 6) -> str:
    sides = int(sides) if sides else 6
    return str(random.randint(1, max(2, sides)))


# --- 文件工具：这里有个必须讲的安全问题 ---
BASE_DIR = Path(__file__).resolve().parent
# 这些目录不能让 agent 翻，否则一次 list 就是几千行
_BLOCKED = {".venv", "__pycache__", ".git", "node_modules"}


def _safe_path(p: str) -> Path:
    """
    把用户/模型给的路径限制在项目目录内，并屏蔽黑名单目录。

    第一层叫「路径穿越防护」：不做的话，模型（或诱导模型的人）传一个
    '../../.env' 就能读到项目外的任何文件 —— 真实存在的安全漏洞。

    第二层是黑名单：显式要求列 '.venv' 也得挡住，
    否则一次 list 就能灌进上千个库文件，瞬间把上下文塞满。
    """
    target = (BASE_DIR / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
    if not (target == BASE_DIR or BASE_DIR in target.parents):
        raise ValueError(f"安全限制：只能访问 {BASE_DIR} 以内的文件")

    rel = target.relative_to(BASE_DIR)
    hit = _BLOCKED.intersection(rel.parts)
    if hit:
        raise ValueError(f"安全限制：目录 {'/'.join(sorted(hit))} 不对 agent 开放")
    return target


def list_files(directory: str = ".") -> str:
    try:
        d = _safe_path(directory or ".")
    except ValueError as e:
        return str(e)
    if not d.exists():
        return f"目录不存在：{directory}"
    if not d.is_dir():
        return f"不是目录：{directory}"

    items = []
    for item in sorted(d.iterdir()):
        if item.name in _BLOCKED or item.name.startswith("."):
            continue
        kind = "目录" if item.is_dir() else "文件"
        size = ""
        if item.is_file():
            kb = item.stat().st_size / 1024
            size = f"  ({kb:.1f} KB)"
        items.append(f"  [{kind}] {item.name}{size}")
    if not items:
        return "目录为空"
    return "\n".join(items[:40]) + ("\n  ...（还有更多）" if len(items) > 40 else "")


def read_file(path: str, max_chars: int = 2000) -> str:
    try:
        f = _safe_path(path)
    except ValueError as e:
        return str(e)

    # 挡掉所有点号开头的隐藏文件（.env / .git / .python-version ...）。
    # 最关键的是 .env —— 里面是 API Key。一旦被模型读走进了 messages，
    # 它就会被原样发到服务器，等于把密钥暴露在对话历史里。
    rel_parts = f.relative_to(BASE_DIR).parts
    if any(part.startswith(".") for part in rel_parts):
        return "安全限制：不能读取隐藏文件"

    if not f.exists():
        return f"文件不存在：{path}"
    if not f.is_file():
        return f"不是文件：{path}"
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"读取失败：{e}"
    max_chars = int(max_chars) if max_chars else 2000
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...（已截断，文件共 {len(text)} 字符）"
    return text


AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "roll_dice": roll_dice,
    "list_files": list_files,
    "read_file": read_file,
}


# ==========================================================================
# 第 3 部分：带重试的请求
# ==========================================================================
def chat(messages, tools=None, retries=6):
    """
    智谱免费档限制同时 1 个请求，连续多步很容易撞 429。
    高峰期（晚上）连续 4 次重试都不够，这里加到 6 次、退避更长，
    总共能熬过约 1 分钟的限流窗口。
    """
    for i in range(retries):
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            wait = 4 * (i + 1)
            err = type(e).__name__
            print(f"      ({err}，等待 {wait}s 后重试 {i + 1}/{retries - 1}，按 Ctrl+C 可放弃)")
            time.sleep(wait)


def execute_tool(name: str, arguments: dict) -> str:
    """执行工具，永远返回字符串 —— 结果要塞回 messages 给模型看。"""
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return f"错误：没有名为 {name} 的工具"
    try:
        return str(func(**arguments))
    except Exception as e:  # noqa: BLE001
        # 注意：工具报错也要把错误信息回给模型。
        # 这样它有机会换个参数重试 —— 这是 agent 能「自我纠错」的关键。
        return f"工具执行出错：{e}"


# ==========================================================================
# 第 4 部分：ReAct 主循环
# ==========================================================================
# 默认值；.env 里配 SYSTEM_PROMPT=xxx 就能覆盖。
# 之所以要能从外部注入：Level 6 做 A/B 实验时，你不想每次改 prompt 都动代码 ——
# 改代码意味着「这次改动到底影响了什么」变得不可控。
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT") or (
    "你是一个会用工具的助手。"
    "可以多次调用工具来收集信息，信息足够了再回答。"
    "不知道的事情要如实说，不要编造。"
)


def run(question: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    total_tokens = 0

    # ---- 这就是 ReAct 的全部 ----
    for step in range(1, MAX_STEPS + 1):
        print(f"\n---- 第 {step} 步 ----")

        response = chat(messages, tools=TOOLS)
        message = response.choices[0].message

        if hasattr(response, "usage") and response.usage:
            total_tokens += response.usage.total_tokens

        # 模型在思考过程中说的话（ReAct 里的 "Reasoning"）。
        # 它可能解释「我先看看有哪些文件」，这就是它的推理痕迹。
        if message.content:
            print(f"  想法: {message.content.strip()[:300]}")

        # ---- 终止条件 1：模型不再要工具了 ----
        if not message.tool_calls:
            print(f"\n===== 最终回答 =====\n{message.content}")
            print(f"\n[共 {step} 步 | 累计 {total_tokens} tokens]")
            return

        # ---- Acting：把模型的请求记进历史，然后执行 ----
        messages.append(message)

        n = len(message.tool_calls)
        print(f"  行动: 请求调用 {n} 个工具")

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)

            preview = result if len(result) <= 200 else result[:200] + " ...(已截断)"
            print(f"    > {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            print(f"      返回: {preview}")

            # 用 tool_call_id 把结果和「哪一次调用」对应起来
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    # ---- 终止条件 2：超过最大步数 ----
    print(f"\n!! 达到最大步数 {MAX_STEPS}，强制结束")
    print(f"[累计 {total_tokens} tokens]")


def main() -> None:
    print("=" * 56)
    print("Level 3：ReAct 循环（输入 q 回车退出，随时可按 Ctrl+C）")
    print("=" * 56)
    print(f"当前厂商 {PROVIDER} | 模型 {MODEL}")
    print(f"接口地址 {BASE_URL}")
    print("试试问：")
    print("  · 项目里有哪些 Python 文件？读一下 main.py 说说是干什么的")
    print("  · 先算 123 乘以 456，再把结果除以 7")
    print("  · 今天星期几？掷个骰子决定我今晚学不学")

    while True:
        try:
            q = input("\n你（输入 q 回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # NFKC 会把全角字符（ｑ）转成半角（q）——
        # 中文输入法状态下敲的 q 常常是全角的，直接比较会匹配失败
        q = unicodedata.normalize("NFKC", q).strip()
        if q.lower() in ("q", "quit", "exit", "退出", "退出退出"):
            break
        if not q:
            continue
        try:
            run(q)
        except KeyboardInterrupt:
            # Ctrl+C 打断一轮长任务，回到输入提示符而不是整个程序崩溃
            print("\n（已打断本轮，回到输入）")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "1305" in msg:
                print("\n限流了：免费档高峰期（尤其晚上）经常这样。")
                print("等一两分钟再试，或者现在就去干别的 —— 这不是你的代码有问题。")
            else:
                print(f"\n出错了：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
