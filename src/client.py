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

    async def run_once(self, image_path: str, user_intent: str = "") -> str:
        async with sse_client(url=self.server_url, headers=self.headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

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

                # 1) 第一次：让上游模型决定调用哪个工具
                resp1 = self.openai.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                msg1 = resp1.choices[0].message

                if not getattr(msg1, "tool_calls", None):
                    return msg1.content or ""

                # 2) 把 assistant(tool_calls) 清洗后放回 messages
                assistant_dict = clean_assistant_tool_calls(msg1)
                messages.append(assistant_dict)

                # 3) 执行工具：注意强制覆盖 image_path，避免模型乱填
                for tc in msg1.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments or "{}")
                    tool_args["image_path"] = image_path  # ✅ 强制覆盖

                    tool_out = await self._call_mcp_tool(session, tool_name, tool_args)
                    print(f"Tool {tool_name} output: {tool_out}")
                    # 云厂商兼容：用 role=function 回传工具结果
                    messages.append(
                        {
                            "role": "function",
                            "name": tool_name,
                            "content": tool_out,
                        }
                    )

                # 4) 第二次：总结（更稳：仍带 tools，并显式 none）
                resp2 = self.openai.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="none",
                )
                return resp2.choices[0].message.content or ""

async def main():
    # ======== 你要的一次性 query：写死在这里 ========
    image_path = "dog1.jpg"
    user_intent = "目标检测,概率越大越好"  # 比如 "目标检测"；留空=让模型自动挑工具

    server_url = "http://localhost:8000/sse"
    model = os.getenv("MODEL", "gpt-4o")

    agent = YOLOMCPAgent(server_url=server_url, model=model)
    result = await agent.run_once(image_path=image_path, user_intent=user_intent)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
