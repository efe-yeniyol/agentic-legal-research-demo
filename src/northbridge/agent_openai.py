"""OpenAI ReAct-style legal research agent over the Northbridge tool set."""

from __future__ import annotations

import json
import os
from typing import Any

from .agent_deterministic import SYSTEM_V1, SYSTEM_V2
from .tools import LegalTools

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the fictional Northbridge statute corpus by free-text query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "Read a Northbridge statute section by id (e.g. NB-CA-12).",
            "parameters": {
                "type": "object",
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_citation",
            "description": "Check whether a claim is supported by a cited Northbridge section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["section_id", "claim"],
            },
        },
    },
]


class OpenAILegalAgent:
    def __init__(self, version: str = "v2", max_steps: int = 6) -> None:
        if version not in {"v1", "v2"}:
            raise ValueError("version must be v1 or v2")
        self.version = version
        self.max_steps = max_steps
        self.system = SYSTEM_V1 if version == "v1" else SYSTEM_V2
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def run(self, prompt: str, force_tool_error: bool = False) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key, "
                "or use method=deterministic."
            )

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        tools = LegalTools(force_error_ids={"NB-XX-99"} if force_tool_error else set())
        steps: list[dict[str, Any]] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]

        final_text = ""
        for step_i in range(1, self.max_steps + 1):
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                final_text = (msg.content or "").strip()
                steps.append(
                    {
                        "step": step_i,
                        "thought": "Model produced a final answer.",
                        "action": "draft",
                        "observation": final_text[:500],
                    }
                )
                break

            assistant_payload: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_payload)

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tools.dispatch(name, args)
                steps.append(
                    {
                        "step": step_i,
                        "thought": f"OpenAI requested tool {name}.",
                        "action": name,
                        "args": args,
                        "observation": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        else:
            # Max steps reached: ask for a final answer without more tools
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop calling tools. Give your final answer now based only on "
                        "observations so far. Do not invent statute text."
                    ),
                }
            )
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
            final_text = (resp.choices[0].message.content or "").strip()
            steps.append(
                {
                    "step": self.max_steps + 1,
                    "thought": "Forced final draft after max steps.",
                    "action": "draft",
                    "observation": final_text[:500],
                }
            )

        if not final_text:
            final_text = "No answer produced."

        return {
            "method": "openai",
            "version": self.version,
            "system_prompt": self.system,
            "mode": "openai_react",
            "model": self.model,
            "output": final_text,
            "steps": steps,
            "n_steps": len(steps),
            "tool_calls": tools.call_log,
            "n_tool_calls": len(tools.call_log),
        }
