"""Level 0 + 1：调通一次 API，然后变成多轮对话。

运行方式（必须在项目目录下）：
    uv run main.py
"""

import os
import unicodedata

from dotenv import load_dotenv
from openai import OpenAI

# 读取 .env 文件里的密钥。这行必须在读 os.environ 之前执行。
load_dotenv()

api_key = os.environ.get("ZAI_API_KEY")

# 友好的报错，避免新手对着一长串 traceback 发懵
if not api_key or "你的" in api_key[:10]:
    print("还没填 API Key。")
    print("请用 VS Code 打开项目根目录下的 .env 文件，")
    print("把 ZAI_API_KEY= 后面换成你自己的密钥，然后重新运行。")
    raise SystemExit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱的 OpenAI 兼容端点
)

# messages 就是「上下文」。每一轮对话都往这个列表里加东西，
# 每次请求都把整个列表发给模型 —— 模型本身不记得你说过什么。
messages = [
    {"role": "system", "content": "你是一个简洁的助手，回答尽量控制在三句话内。"},
]

print("开始对话，输入 q 回车退出（随时可按 Ctrl+C）。\n")

while True:
    try:
        q = input("你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见。")
        break

    # NFKC 把全角 ｑ 转成半角 q（中文输入法下容易敲出全角）
    q = unicodedata.normalize("NFKC", q).strip()

    if not q:
        continue
    if q.lower() in ("q", "quit", "exit", "退出"):
        print("再见。")
        break

    # 把用户的话加进历史
    messages.append({"role": "user", "content": q})

    try:
        response = client.chat.completions.create(
            model="glm-4.7-flash",  # 智谱免费档，支持工具调用，学习调试足够
            messages=messages,  # 注意：是整个历史，不是只有这一句
        )
    except KeyboardInterrupt:
        # 请求还没回来就被打断了：把刚加进历史的话撤掉，保持上下文干净
        messages.pop()
        print("（已打断，这条没发出去）")
        continue

    answer = response.choices[0].message.content
    print("AI:", answer)

    # 把模型的回复也加进历史，否则下一轮它就忘了自己说过什么
    messages.append({"role": "assistant", "content": answer})

    print(f"[累计 {len(messages)} 条消息 | 本次消耗 {response.usage.total_tokens} tokens]\n")
