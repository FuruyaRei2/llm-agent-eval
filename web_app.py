"""
Level 5（第二部分）：网页界面

终端里看流式不过瘾，做成网页才算一个「产品」。

这里用的是 SSE（Server-Sent Events）：
    浏览器发一个普通 GET 请求，服务器不一次性返回，
    而是把响应体保持打开，有事件就往里写一行：

        data: {"type":"answer_delta","text":"你"}

        data: {"type":"answer_delta","text":"好"}

    （每个事件之间空一行，这是 SSE 的格式规定）

为什么不用 WebSocket？
    SSE 是单向的（服务器 -> 浏览器），但agent 对话正好只需要这个方向：
    用户发一句话是一次普通请求，之后全是服务器往外推。
    SSE 基于普通 HTTP，浏览器原生支持（EventSource），不用装任何库，
    还能自动断线重连。WebSocket 是双向的，这里杀鸡用牛刀了。

对比一下三种做法，面试常问：
    · 普通请求       简单，但必须等全部完成，用户盯着白屏
    · SSE           单向推送，HTTP 原生，适合 agent 逐步输出
    · WebSocket     双向实时，适合协同编辑/多人游戏，这里用不上

运行：
    uv run web_app.py
然后浏览器打开 http://127.0.0.1:8000
"""

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from level3_react import BASE_URL, MODEL, PROVIDER
from level4_trace import TraceLogger
from level5_stream import run_stream

app = FastAPI()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/chat")
def chat(q: str = Query(..., min_length=1)):
    """
    SSE 接口。返回一个生成器，FastAPI 会把它变成一条「永不完全结束」的响应。

    注意这个生成器是同步的（def 不是 async def）。
    FastAPI 会自动把它放到线程池里跑，所以里面的 time.sleep()（限流退避）
    不会卡住整个服务器 —— 这点很重要，否则一个人被限流，所有人一起卡。
    """

    def event_stream():
        logger = TraceLogger()
        try:
            for ev in run_stream(q, logger):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "1305" in msg:
                msg = "被限流了。免费档同时只允许 1 个请求，等 30 秒再试。"
            yield "data: " + json.dumps(
                {"type": "error", "message": f"{type(e).__name__}: {msg}"},
                ensure_ascii=False,
            ) + "\n\n"
        # 告诉前端「真的结束了」，让它断开 EventSource
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 关掉 nginx 一类反向代理的缓冲，否则流式会被攒成一坨
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    print("Level 5 网页界面启动中...")
    print(f"厂商 {PROVIDER} | 模型 {MODEL}")
    print(f"接口 {BASE_URL}")
    print("浏览器打开： http://127.0.0.1:8000")
    print("（Ctrl+C 停止）")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
