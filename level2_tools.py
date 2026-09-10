"""
Level 2：让模型学会「主动调用工具」

Level 0+1 里模型只能说话。这一级加一件关键的事：
告诉模型「你有哪些工具可用」，模型会自己判断要不要用、用哪个、传什么参数。

你会第一次看到模型回复里出现 tool_calls —— 它不是在给你答案，
而是在说「请帮我执行这个工具，然后把结果告诉我」。

运行：
    uv run level2_tools.py
试试问：
    123 * 456 等于多少
    现在几点了
    帮我掷一个 20 面骰子
    我今天该带伞吗        ← 没有对应工具，模型会自己回答
"""

import ast
import json
import operator
import os
import random
import time
import unicodedata
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/furuyarei/projects/llm-agent/.env")

api_key = os.environ.get("ZAI_API_KEY")
if not api_key or len(api_key) < 20:
    print("还没填 API Key，去 .env 里把 ZAI_API_KEY 换成你的智谱密钥。")
    raise SystemExit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

MODEL = "glm-4.7-flash"


# --------------------------------------------------------------------------
# 第 1 部分：工具的「说明书」
#
# 这段 JSON 是写给模型看的。模型不会真的执行任何东西，
# 它只是根据这份说明书，决定「该调谁、参数填什么」。
# description 写得越清楚，模型用得越准 —— 这是 prompt engineering 的一部分。
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。当用户问算术、百分比、单位换算等需要精确结果的问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '123 * 456' 或 '(88 + 12) / 4'",
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
            "description": "获取当前的日期和时间。当用户问现在几点、今天几号、星期几时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "掷骰子，返回一个随机整数。模型自己无法产生真正的随机数，必须用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sides": {
                        "type": "integer",
                        "description": "骰子的面数，默认 6",
                    }
                },
            "required": [],
            },
        },
    },
]


# --------------------------------------------------------------------------
# 第 2 部分：工具的「真实实现」
#
# 上面是说明书，这里才是真正跑的代码。两边通过函数名（calculator 等）对上。
# --------------------------------------------------------------------------

# 只允许这些运算，避免 eval() 的安全问题
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
    """只解析数字和四则运算的语法树，遇到其它东西（函数调用、变量名）就报错。"""
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
    """安全版计算器 —— 注意这里绝不能用 eval()，那是严重的安全漏洞。"""
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
        return str(value)
    except Exception as e:  # noqa: BLE001
        return f"计算失败：{e}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def roll_dice(sides: int = 6) -> str:
    sides = int(sides) if sides else 6
    return str(random.randint(1, max(2, sides)))


# 函数名 -> 真实函数。模型只给名字，我们靠这个表找到要执行的代码。
AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "roll_dice": roll_dice,
}


def chat(messages, tools=None, retries=5):
    """
    带重试的请求。

    为什么需要：智谱免费档限制「同时 1 个请求」，连续两次调用很容易撞上
    429（访问量过大 / RateLimitError）。重试 + 退避等待是 agent 的必备处理。
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
            wait = 2 * (i + 1)
            print(f"    (遇到 {type(e).__name__}，等待 {wait}s 后重试 {i + 1}/{retries - 1})")
            time.sleep(wait)


def execute_tool(name: str, arguments: dict) -> str:
    """执行一个工具。永远返回字符串 —— 因为要塞回 messages 里给模型看。"""
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return f"错误：没有名为 {name} 的工具"
    try:
        return str(func(**arguments))
    except Exception as e:  # noqa: BLE001
        return f"工具执行出错：{e}"


# --------------------------------------------------------------------------
# 第 3 部分：主流程
# --------------------------------------------------------------------------
def ask(question: str) -> None:
    messages = [{"role": "user", "content": question}]

    # --- 第一次请求：把工具说明书一起交给模型 ---
    print("\n[1] 发给模型：问题 + 3 个工具的说明书")
    response = chat(messages, tools=TOOLS)
    message = response.choices[0].message

    # --- 模型可能回两种东西：普通回答，或者工具调用请求 ---
    if not message.tool_calls:
        print(f"\n[2] 模型认为不需要工具，直接回答：\n    {message.content}")
        return

    # --- 模型要调工具了，打印出来看看它到底说了什么 ---
    print(f"\n[2] 模型没有直接回答，而是请求调用工具：")
    for tc in message.tool_calls:
        print(f"    工具名: {tc.function.name}")
        print(f"    参数:   {tc.function.arguments}")
    print(f"    (模型自己说的话: {message.content!r})")

    # 关键一步：把「模型要求调工具」这条消息也放进历史，
    # 否则模型后面会忘记自己刚才要干什么。
    messages.append(message)

    # --- 执行工具，把结果按顺序回灌 ---
    print("\n[3] 我们执行工具：")
    for tc in message.tool_calls:
        args = json.loads(tc.function.arguments)
        result = execute_tool(tc.function.name, args)
        print(f"    {tc.function.name}({args}) -> {result}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,  # 这个 id 用来对应「哪次调用的结果」
                "content": result,
            }
        )

    # --- 第二次请求：模型看到工具结果，组织成自然语言回答 ---
    print("\n[4] 把结果交还给模型，让它组织最终回答")
    final = chat(messages, tools=TOOLS)
    print(f"\n[5] 最终回答：\n    {final.choices[0].message.content}")


def main() -> None:
    print("=" * 52)
    print("Level 2：工具调用（输入 q 退出）")
    print("=" * 52)
    print("可以试试：123 * 456 等于多少 / 现在几点 / 掷个 20 面骰子")

    while True:
        try:
            q = input("\n你（输入 q 回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # NFKC 把全角 ｑ 转成半角 q（中文输入法状态下很容易敲出全角）
        q = unicodedata.normalize("NFKC", q).strip()
        if q.lower() in ("q", "quit", "exit", "退出"):
            break
        if not q:
            continue
        try:
            ask(q)
        except KeyboardInterrupt:
            print("\n（已打断本轮，回到输入）")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "1305" in msg:
                print("\n限流了：免费档高峰期很常见，等一两分钟再试。")
            else:
                print(f"\n出错了：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
