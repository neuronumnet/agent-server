import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Model backend (OpenAI-compatible) ---------------------------------------
#
# No provider is baked into the image. Point MODEL_BASE_URL + CLIENT_API_KEY
# + MODEL_NAME at any OpenAI-compatible endpoint: OpenAI itself, Groq,
# OpenRouter, a self-hosted Ollama/llama.cpp server, etc.

if not os.environ.get("CLIENT_API_KEY"):
    raise SystemExit(
        "CLIENT_API_KEY is not set. neuronum-server needs an OpenAI-compatible "
        "model endpoint to run — set CLIENT_API_KEY (and optionally "
        "MODEL_BASE_URL / MODEL_NAME / MAX_TOKENS / TEMPERATURE) in .env. "
        "See README for provider examples."
    )

client = OpenAI(
    base_url=os.environ.get("MODEL_BASE_URL") or None,
    api_key=os.environ["CLIENT_API_KEY"],
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))


def call_model(system: str, history: list[dict], tools: list[dict] | None = None) -> dict:
    request = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": system}] + history,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    response = client.chat.completions.create(**request)
    message = response.choices[0].message

    if message.tool_calls:
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
        return {
            "_assistant_message": assistant_message,
            "_tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                for tool_call in message.tool_calls
            ],
        }

    raw = (message.content or "").strip()
    return _parse_response(raw)


def _parse_response(raw: str) -> dict:
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"msg": raw}
        if isinstance(parsed.get("msg"), str) and not parsed.get("element"):
            inner = parsed["msg"].strip()
            if inner.startswith("{"):
                try:
                    inner_parsed = json.loads(inner)
                    if isinstance(inner_parsed, dict) and inner_parsed.get("element"):
                        return inner_parsed
                except json.JSONDecodeError:
                    pass
        return parsed
    except (json.JSONDecodeError, IndexError):
        return {"msg": raw}
