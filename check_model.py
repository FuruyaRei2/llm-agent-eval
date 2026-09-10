"""
模型体检脚本：换厂商/换模型之后先跑它，回答三个问题
  1. 我现在的 key 连的是哪一家、哪个模型？
  2. 我的账号能调用哪些模型？（实测，不是查文档）
  3. 余额还有多少？（付费厂商才有这个接口）

用法：
    uv run check_model.py                # 测当前 .env 配出来的模型
    uv run check_model.py deepseek-v4-pro  # 临时测另一个模型

注意：本脚本只打印模型名和余额，不会打印你的 API Key。
"""

import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# 复用 level3 的配置逻辑 —— 保证「体检脚本」和「真正跑的」永远是同一套配置，
# 不会出现「体检通过但实际跑的是另一家」这种错觉
from level3_react import BASE_URL, MODEL, PROVIDER, api_key  # noqa: E402

# 智谱官方公开标注为「免费」的模型（2026-09 核实，随时可能变，以实测为准）
KNOWN_FREE = {
    "glm-4.7-flash", "glm-4.5-flash", "glm-4-flash-250414",
    "glm-4.6v-flash", "glm-4.1v-thinking-flash", "glm-4v-flash",
    "cogview-3-flash", "cogvideox-flash",
}

headers = {"Authorization": f"Bearer {api_key}"}


def show_balance() -> None:
    """
    DeepSeek 提供 /user/balance 查询余额（智谱没有这个接口）。
    这是付费之后最该盯的数字 —— 别等跑不出来了才发现余额归零。
    """
    if "DEEPSEEK" not in PROVIDER.upper():
        return
    print()
    print("=" * 56)
    print("第 0 步：DeepSeek 账户余额")
    print("=" * 56)
    try:
        r = httpx.get(f"{BASE_URL}/user/balance", headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"  查询失败 HTTP {r.status_code}：{r.text[:200]}")
            return
        data = r.json()
        infos = data.get("balance_infos", [])
        if not infos:
            print(f"  返回内容：{data}")
            return
        for b in infos:
            cur = b.get("currency", "")
            print(f"  {cur} 总余额 {b.get('total_balance')}"
                  f"（充值 {b.get('topped_up_balance')} / 赠送 {b.get('granted_balance')}）")
        print(f"  是否可用：{data.get('is_available')}")
    except Exception as e:  # noqa: BLE001
        print(f"  查询余额失败（不影响后续）：{e}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else MODEL

    print("=" * 56)
    print("当前配置")
    print("=" * 56)
    print(f"  厂商    {PROVIDER}")
    print(f"  模型    {target}")
    print(f"  接口    {BASE_URL}")
    print(f"  Key     {'已配置（' + str(len(api_key)) + ' 位，不显示内容）'}")

    show_balance()

    # ---------- 第 1 步：列出账号可用的模型 ----------
    print()
    print("=" * 56)
    print("第 1 步：你的账号能调用哪些模型")
    print("=" * 56)
    try:
        resp = httpx.get(f"{BASE_URL}/models", headers=headers, timeout=30)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        models.sort()
        print(f"共 {len(models)} 个：\n")
        for mid in models:
            tag = "  ← 免费档" if mid in KNOWN_FREE else ""
            print(f"  {mid}{tag}")
    except Exception as e:  # noqa: BLE001
        print(f"  拉取模型列表失败：{e}")
        print("  （不影响下一步，继续实测指定模型）")

    # ---------- 第 2 步：实测目标模型 ----------
    print()
    print("=" * 56)
    print(f"第 2 步：实测 {target}")
    print("=" * 56)
    payload = {
        "model": target,
        "messages": [{"role": "user", "content": "只回复两个字：可用"}],
        "max_tokens": 32,
    }
    try:
        r = httpx.post(
            f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60
        )
    except Exception as e:  # noqa: BLE001
        print(f"  请求失败：{e}")
        raise SystemExit(1) from e

    if r.status_code != 200:
        print(f"  HTTP {r.status_code}")
        print(f"  {r.text[:400]}")
        print("\n  常见原因：")
        print("   · 402 Insufficient Balance —— 余额不足，去 platform.deepseek.com 充值")
        print("   · 401 —— key 无效或复制时带上了空格")
        print("   · 404 —— 模型名写错（注意是 deepseek-v4-flash，不是 deepseek-chat）")
        raise SystemExit(1)

    data = r.json()
    msg = data["choices"][0]["message"]
    print(f"  模型实际返回：{msg.get('content')!r}")
    if msg.get("reasoning_content"):
        print(f"  思考过程：{msg['reasoning_content'][:80]}…（说明这是推理模型）")
    print(f"  token 用量：{data.get('usage', {})}")

    # ---------- 第 3 步：工具调用是不是真的支持 ----------
    # 能聊天不等于能当 agent 用 —— 必须单独验证 function calling
    print()
    print("=" * 56)
    print("第 3 步：实测工具调用（agent 的命根子）")
    print("=" * 56)
    tool_payload = {
        "model": target,
        "messages": [{"role": "user", "content": "123 乘以 456 等于多少？用工具算"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    try:
        r2 = httpx.post(
            f"{BASE_URL}/chat/completions", headers=headers, json=tool_payload, timeout=60
        )
        if r2.status_code == 200:
            tcs = r2.json()["choices"][0]["message"].get("tool_calls")
            if tcs:
                print(f"  支持。返回了工具调用：{tcs[0]['function']['name']}"
                      f"({tcs[0]['function']['arguments']})")
            else:
                print("  返回了 200 但没有 tool_calls —— 该模型可能不支持/未开启工具调用")
        else:
            print(f"  HTTP {r2.status_code}：{r2.text[:200]}")
    except Exception as e:  # noqa: BLE001
        print(f"  工具调用测试失败：{e}")


if __name__ == "__main__":
    main()
