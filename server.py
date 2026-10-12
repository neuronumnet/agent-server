import asyncio
import json
import re
from urllib.parse import quote, urlsplit

import aiohttp
import jsonschema

from neuronum import AgentIdentity
from model import call_model
from elements import validate_element_payload

instruction_cache: dict[str, str] = {}
metadata_cache: dict[str, dict] = {}
session_started: set[str] = set()
MAX_TOOL_CALL_ROUNDS = 5
MAX_TOOL_CALLS_PER_ROUND = 5
TOOL_REQUEST_TIMEOUT_SECONDS = 20
MAX_TOOL_RESPONSE_BYTES = 32768
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TOOL_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


SYSTEM_PROMPT = """
## Identity

You are a conversational agent speaking with customers or business partners. You guide the conversation dynamically based on your instructions, gathering and exchanging information naturally step by step.

## Instruction

## Tools

When configured tools can provide relevant current or external information, use them instead of guessing. Only use the tools provided to you, and base your answer on their results.


## Response format

You must always reply with a single JSON object. The outer response must be valid JSON — no markdown or prose outside it. Inside the "msg" value, markdown is fully supported and encouraged for formatting text responses.

The "msg" key is always required. The "element" key is optional — only include it when an interactive UI component genuinely improves the experience over plain text.

## Elements

Plain message (default — use for most replies):
{"msg": "your text"}

Confirm — binary yes/no decision. Use when the user must explicitly accept or reject something before proceeding:
{"msg": "Do you approve?", "element": "confirm"}

Choice — pick one from a fixed set of options. Use when options are known upfront and mutually exclusive:
{"msg": "Which plan suits you best?", "element": "choice", "choices": ["Starter", "Pro", "Enterprise"]}

Input — collect one specific labelled value. Use only for structured data collection, not for general chat:
{"msg": "What is your company name?", "element": "input", "placeholder": "Acme Corp"}

Form — collect multiple specific labelled values in one step. Use instead of asking questions one by one:
{"msg": "Tell us about yourself:", "element": "form", "fields": [{"name": "company", "label": "Company", "placeholder": "Acme Corp"}, {"name": "role", "label": "Role", "placeholder": "CEO"}]}

Table — present structured data clearly. "columns" must be a flat list of strings. "rows" must be a list of lists of strings:
{"msg": "Here is your summary:", "element": "table", "columns": ["Field", "Value"], "rows": [["Company", "Acme"], ["Role", "CEO"]]}

Card — combine multiple elements into one message. Use when you need to show a summary and collect a response in the same step:
{"msg": "Review and confirm:", "element": "card", "components": [{"type": "table", "columns": ["Field", "Value"], "rows": [["Company", "Acme"], ["Role", "CEO"]]}, {"type": "confirm", "name": "approved", "label": "Does this look correct?"}]}

Link — a clickable button that opens a URL. Use for payments, external pages, or next-step actions:
{"msg": "Complete your payment", "element": "link", "link": "https://checkout.stripe.com/pay/cs_live_abc123"}

File — prompt the user to upload a file:
{"msg": "Please upload your signed contract:", "element": "file"}

## Rules

- Never send a table followed immediately by a confirm as two separate messages. Use a card instead.
- Do not use "input" or "form" for general conversation — the user already has a chat input. Reserve them for collecting specific structured data.
- Do not use "choice" for open-ended questions — only when the full set of options is known.
- All row values in tables must be strings, not numbers.
- If no element genuinely improves the interaction, omit "element" entirely.
"""


async def get_session_context(identity: AgentIdentity, session_id: str) -> str:
    """Returns full system prompt built from session metadata. Caches per session."""
    if session_id not in instruction_cache:
        metadata = await identity.fetch_session_metadata(session_id)
        metadata_cache[session_id] = metadata or {}
        instruct = metadata.get("instruct", "") if metadata else ""
        instruction_cache[session_id] = SYSTEM_PROMPT.replace("## Instruction\n", f"## Instruction\n\n{instruct}\n")
    return instruction_cache[session_id]


async def build_history(identity: AgentIdentity, session_id: str) -> list[dict]:
    messages = await identity.get_session_messages(session_id)
    history = []
    for m in messages:
        data = m["data"]
        action = data.get("action")
        if action == "session_joined":
            continue
        text = data.get("msg", "")
        element = data.get("element")
        if action == "element_response" and element and isinstance(text, dict):
            text = f"[{element} response] {json.dumps(text)}"
        elif action == "element_response" and element and isinstance(text, str):
            text = f"[{element} response] {text}"
        elif isinstance(text, dict):
            text = json.dumps(text)
        elif not isinstance(text, str):
            text = str(text)
        if not text:
            continue
        role = "assistant" if m["sender"] == identity.host else "user"
        history.append({"role": role, "content": text})
    return history


def get_session_tools(session_id: str) -> list[dict]:
    tools = metadata_cache.get(session_id, {}).get("tools", [])
    return tools if isinstance(tools, list) else []


def build_model_tools(session_tools: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    model_tools = []
    configured_tools = {}

    for tool in session_tools:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name")
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        if (
            not isinstance(name, str)
            or not TOOL_NAME_PATTERN.fullmatch(name)
            or name in configured_tools
            or not isinstance(parameters, dict)
        ):
            continue

        schema = dict(parameters)
        schema.setdefault("type", "object")
        if schema["type"] != "object":
            continue

        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError:
            continue

        function = {"name": name, "parameters": schema}
        description = tool.get("description")
        if isinstance(description, str) and description:
            function["description"] = description

        model_tools.append({"type": "function", "function": function})
        configured_tools[name] = tool

    return model_tools, configured_tools


async def execute_tool_call(
    session: aiohttp.ClientSession,
    tool: dict,
    raw_arguments: str,
) -> dict:
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    schema = tool.get("parameters") or {"type": "object", "properties": {}}
    jsonschema.validate(arguments, schema)

    url = tool.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Configured tool URL is missing.")

    used_path_arguments = set()

    def replace_path_parameter(match: re.Match) -> str:
        name = match.group(1)
        if name not in arguments:
            raise ValueError(f"Missing URL path argument: {name}")
        value = arguments[name]
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"URL path argument '{name}' must be a scalar value.")
        used_path_arguments.add(name)
        return quote(str(value), safe="")

    url = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_path_parameter, url)
    if "{" in url or "}" in url:
        raise ValueError("Tool URL contains an invalid path parameter.")

    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.fragment
    ):
        raise ValueError("Tool URL must be an HTTP(S) URL without credentials or a fragment.")

    method = tool.get("method", "GET")
    if not isinstance(method, str) or method.upper() not in TOOL_METHODS:
        raise ValueError("Tool HTTP method is not supported.")
    method = method.upper()

    auth = tool.get("auth") or {}
    if not isinstance(auth, dict):
        raise ValueError("Tool auth configuration must be an object.")

    headers = {}
    bearer = auth.get("bearer")
    if bearer is not None:
        if not isinstance(bearer, str):
            raise ValueError("Tool bearer token must be a string.")
        headers["Authorization"] = f"Bearer {bearer}"

    extra_headers = auth.get("headers") or {}
    if not isinstance(extra_headers, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in extra_headers.items()
    ):
        raise ValueError("Tool auth headers must be string key/value pairs.")
    headers.update(extra_headers)

    remaining_arguments = {
        key: value for key, value in arguments.items() if key not in used_path_arguments
    }
    request_options = {"headers": headers, "allow_redirects": False}
    if method in {"GET", "DELETE"}:
        request_options["params"] = remaining_arguments
    else:
        request_options["json"] = remaining_arguments

    async with session.request(method, url, **request_options) as response:
        body = bytearray()
        async for chunk in response.content.iter_chunked(8192):
            body.extend(chunk)
            if len(body) > MAX_TOOL_RESPONSE_BYTES:
                raise ValueError("Tool response exceeded the 32 KiB limit.")

        text = body.decode("utf-8", errors="replace")
        try:
            result = json.loads(text) if text else None
        except json.JSONDecodeError:
            result = text

        return {
            "status_code": response.status,
            "success": 200 <= response.status < 300,
            "data": result,
        }


def format_reply(reply: dict) -> dict:
    try:
        return validate_element_payload(reply)
    except ValueError:
        return {"msg": reply.get("msg", str(reply))}


async def generate_reply(system: str, history: list[dict], session_tools: list[dict]) -> dict:
    model_tools, configured_tools = build_model_tools(session_tools)
    conversation = list(history)

    if not model_tools:
        reply = await asyncio.to_thread(call_model, system, conversation)
        return format_reply(reply)

    timeout = aiohttp.ClientTimeout(total=TOOL_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        for _ in range(MAX_TOOL_CALL_ROUNDS):
            reply = await asyncio.to_thread(call_model, system, conversation, model_tools)
            tool_calls = reply.get("_tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                return format_reply(reply)

            assistant_message = reply.get("_assistant_message")
            if not isinstance(assistant_message, dict):
                return {"msg": "I couldn't process the requested tool call. Please try again."}
            conversation.append(assistant_message)

            for index, tool_call in enumerate(tool_calls):
                call_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                if index >= MAX_TOOL_CALLS_PER_ROUND:
                    result = {"error": "Too many tool calls were requested at once."}
                elif tool_name not in configured_tools:
                    result = {"error": "The requested tool is not configured for this session."}
                else:
                    try:
                        result = await execute_tool_call(
                            http_session,
                            configured_tools[tool_name],
                            tool_call.get("arguments", "{}"),
                        )
                    except (json.JSONDecodeError, jsonschema.ValidationError, ValueError) as exc:
                        result = {"error": str(exc)}
                    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        print(f"[worker] Tool {tool_name} request failed: {type(exc).__name__}")
                        result = {"error": f"The tool request failed ({type(exc).__name__})."}
                    except Exception as exc:
                        print(f"[worker] Tool {tool_name} failed: {type(exc).__name__}")
                        result = {"error": "The tool failed unexpectedly."}

                conversation.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

    return {"msg": "I couldn't finish the requested tool operations. Please try again."}


async def listen(identity: AgentIdentity):
    async for message in identity.sync_messages():
        session_id = message["session_id"]
        sender = message["sender"]
        data = message["data"]
        action = data.get("action")
        user_text = data.get("msg", "")

        if sender == identity.host:
            continue

        # ── Session joined — start conversation immediately ───────────────────
        if action == "session_joined":
            if session_id in session_started:
                continue
            try:
                existing = await identity.get_session_messages(session_id)
                if any(m["sender"] == identity.host for m in existing):
                    session_started.add(session_id)
                    continue
            except Exception:
                pass
            session_started.add(session_id)
            try:
                system = await get_session_context(identity, session_id)
                greeting = await generate_reply(
                    system,
                    [{"role": "user", "content": "Introduce yourself briefly and let me know how you can help."}],
                    get_session_tools(session_id),
                )
                await identity.send_session_message(session_id, greeting)
            except Exception as e:
                print(f"[worker] Error sending greeting in {session_id}: {e}")
            continue

        # ── Element response ──────────────────────────────────────────────────
        if action == "element_response":
            try:
                system = await get_session_context(identity, session_id)
                history = await build_history(identity, session_id)
                reply = await generate_reply(system, history, get_session_tools(session_id))
                await identity.send_session_message(session_id, reply)
            except Exception as e:
                print(f"[worker] Error on element_response in {session_id}: {e}")
            continue

        # ── File uploads ──────────────────────────────────────────────────────
        if action == "file":
            continue

        # ── Regular message ───────────────────────────────────────────────────
        if not user_text:
            continue

        try:
            system = await get_session_context(identity, session_id)
            history = await build_history(identity, session_id)
            reply = await generate_reply(system, history, get_session_tools(session_id))
            await identity.send_session_message(session_id, reply)
        except Exception as e:
            print(f"[worker] Error handling message in {session_id}: {e}")


async def main():
    retry_delay = 2

    async with AgentIdentity() as identity:
        while True:
            try:
                await listen(identity)
            except Exception as e:
                print(f"[worker] SSE error: {e}")
            await asyncio.sleep(retry_delay)


asyncio.run(main())
