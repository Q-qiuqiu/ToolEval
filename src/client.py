from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI

load_dotenv()


def mcp_tool_to_openai_tool(tool: Any) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }

def clean_assistant_tool_calls(msg1) -> Dict[str, Any]:
    """
    把第一次 LLM 返回的 assistant message 清洗成更通用的 OpenAI-compatible 格式：
    - 强制带 content 字段（某些后端要求）
    - tool_calls 去掉 index / 多余字段
    """
    assistant_dict = msg1.model_dump(exclude_none=True)
    assistant_dict["content"] = assistant_dict.get("content", "") or ""

    if "tool_calls" in assistant_dict:
        cleaned = []
        for tc in assistant_dict["tool_calls"]:
            cleaned.append(
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
            )
        assistant_dict["tool_calls"] = cleaned
    return assistant_dict

class YOLOMCPAgent:
    def __init__(self, server_url: str, model: str = "gpt-4o-mini", headers: Optional[Dict[str, str]] = None):
        self.server_url = server_url
        self.headers = headers or {}
        self.openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "empty"),  # 有些本地服务不校验，但 openai SDK 需要给个值
            base_url=os.getenv("BASE_URL", "http://localhost:1234/v1"),
        )
        self.llm_model = model

    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, tool_args: Dict[str, Any]) -> str:
        result = await session.call_tool(tool_name, tool_args)

        # 把 MCP 的 TextContent 提取成纯文本（更利于 LLM 总结）
        if hasattr(result, "content") and result.content:
            texts = []
            for c in result.content:
                t = getattr(c, "text", None)
                if t is not None:
                    texts.append(t)
            if texts:
                return "\n".join(texts)

        # 兜底
        try:
            return json.dumps(result.content, ensure_ascii=False, default=str)
        except Exception:
            return str(result)

    async def run_once(self, session: ClientSession, image_path: str, user_intent: str = "") -> str:
        tools_resp = await session.list_tools()

        openai_tools = [mcp_tool_to_openai_tool(t) for t in tools_resp.tools]

        system = (
            "你是一个视觉推理助手。你可以调用工具对图片做目标检测。"
            "请根据用户意图选择最合适的检测工具并调用。"
            "输出时先给出简短结论，再给出 JSON 结果。"
        )
        user = (
            f"图片路径：{image_path}\n"
            f"用户意图：{user_intent or '无（请自动选择最合适模型）'}\n"
            "请调用一个合适的 yolo_detect_* 工具对该图片做检测。"
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # 1) 让 LLM 决定是否调用工具
        resp = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        msg1 = resp.choices[0].message

        # 没有 tool_calls 就直接返回
        if not getattr(msg1, "tool_calls", None):
            return msg1.content or ""
        assistant_dict = msg1.model_dump(exclude_none=True)
        assistant_dict["content"] = assistant_dict.get("content", "") or ""
        if "tool_calls" in assistant_dict:
            cleaned = []
            for tc in assistant_dict["tool_calls"]:
                cleaned.append({
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                })
            assistant_dict["tool_calls"] = cleaned
        messages.append(assistant_dict)

        for tc in msg1.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments or "{}")
            tool_args.setdefault("image_path", image_path)

            tool_out = await self._call_mcp_tool(session,tool_name, tool_args)

            messages.append({
                "role": "function",
                "name": tool_name,   # 注意是函数名，如 detect_yolov8n
                "content": tool_out,
            })
        #print(json.dumps(messages, ensure_ascii=False, indent=2))
        # 第二次 LLM：总结
        resp2 = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tool_choice="none", 
        )

        return resp2.choices[0].message.content or ""
        
    async def chat_loop(self) -> None:
        # 推荐写法：用 async with 管理连接生命周期 :contentReference[oaicite:1]{index=1}
        async with sse_client(url=self.server_url, headers=self.headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                #print("[Connected tools]", [t.name for t in tools.tools])

                print("\n输入图片路径开始推理；输入 quit 退出。")
                while True:
                    image_path = input("\nimage_path> ").strip()
                    if image_path.lower() in {"q", "quit", "exit"}:
                        break
                    user_intent = input("intent (可选，比如'安全帽检测' / 回车自动)> ").strip()

                    try:
                        out = await self.run_once(session=session, image_path=image_path, user_intent=user_intent)
                        print("\n=== RESULT ===\n", out)
                    except Exception as e:
                        print("\n[Error]", repr(e))


async def main():
    server_url = "http://localhost:8000/sse"
    model = os.getenv("MODEL", "gpt-4o")

    agent = YOLOMCPAgent(server_url=server_url, model=model)
    await agent.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
