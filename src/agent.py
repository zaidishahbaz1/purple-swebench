"""
SWE-bench Pro Purple Agent — single-shot patch generation.

Protocol (SWE-bench Pro green agent expects one response per task):
  Green -> Purple: {"instance_id", "problem_statement", "image_uri", "base_commit", "repo", "hints"}
  Purple -> Green: response containing a unified diff. We return {"patch": "<diff>"}.

The green agent's _extract_patch tries several formats; we use the JSON form
{"patch": "..."} which is the most robust.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re

PROBLEM_MAX_CHARS = 4000
HINTS_MAX_CHARS = 1000
RETRY_MAX_ATTEMPTS = 15
RETRY_BACKOFF_CAP = 30  # seconds

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

import litellm
from dotenv import load_dotenv

from messenger import Messenger

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert software engineer.

You will be given a GitHub issue from a real open-source repository. Your job is to write a unified-diff patch that fixes the issue. You have prior knowledge of the repository's code structure but you cannot run commands or read files directly — produce the best patch you can from the issue alone, using your training knowledge of how that repository is typically organized.

You MUST respond with a single JSON object of this exact shape:
  {"patch": "<unified diff>"}

The "patch" string must be a valid unified diff (the kind `git apply` accepts), with file headers and hunks. For example:

{"patch": "diff --git a/path/to/file.py b/path/to/file.py\\n--- a/path/to/file.py\\n+++ b/path/to/file.py\\n@@ -10,7 +10,7 @@\\n context\\n-old line\\n+new line\\n context\\n"}

Critical rules:
- Output ONLY the JSON object. No prose, no markdown fences, no explanation.
- Use real file paths consistent with the named repository.
- Provide enough context lines (3 above + 3 below changes) for the hunk to apply.
- Do not break existing tests. Make the minimal change.
- NEVER return an empty patch. You MUST attempt a fix even when uncertain. A wrong-but-plausible patch that touches the right area is strictly better than no patch — wrong patches sometimes accidentally pass tests, and at worst score the same as empty.
- Reason through the issue: identify the symptom, hypothesize the buggy file/function based on the repo's conventional layout, and propose a minimal change that addresses the symptom. Pick a concrete file path and produce the diff even if your confidence is low.
"""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text) - limit} chars ...]"


def _build_user_prompt(task: dict) -> str:
    problem = _truncate(task.get("problem_statement", "").strip(), PROBLEM_MAX_CHARS)
    hints = _truncate(
        (task.get("hints") or task.get("hints_text") or "").strip(),
        HINTS_MAX_CHARS,
    )
    repo = task.get("repo", "").strip()
    base_commit = task.get("base_commit", "").strip()

    parts = []
    if repo:
        parts.append(f"Repository: {repo}")
    if base_commit:
        parts.append(f"Base commit: {base_commit}")
    parts.append("")
    parts.append("## Problem statement")
    parts.append("")
    parts.append(problem)
    if hints:
        parts.append("")
        parts.append("## Hints (issue thread context)")
        parts.append("")
        parts.append(hints)
    parts.append("")
    parts.append("Now produce the patch as JSON: {\"patch\": \"<unified diff>\"}.")
    return "\n".join(parts)


def _extract_patch_from_response(response: str) -> str:
    """Pull a usable unified diff out of the model's response."""
    if not response:
        return ""
    response = response.strip()

    # 1. Try parsing as full JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "patch" in data:
            patch = data["patch"]
            if isinstance(patch, str):
                return patch
    except json.JSONDecodeError:
        pass

    # 2. Try parsing JSON wrapped in ```...``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict) and "patch" in data:
                return data["patch"] or ""
        except json.JSONDecodeError:
            pass

    # 3. Try parsing first JSON object substring
    obj_match = re.search(r"\{.*\}", response, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict) and "patch" in data:
                return data["patch"] or ""
        except json.JSONDecodeError:
            pass

    # 4. Markdown diff fence
    diff_fence = re.search(r"```(?:diff|patch)?\s*\n(diff --git .*?)\n```", response, re.DOTALL)
    if diff_fence:
        return diff_fence.group(1)

    # 5. Raw diff anywhere in the text
    raw = re.search(r"(diff --git [\s\S]+)$", response)
    if raw:
        return raw.group(1)

    return ""


def _normalize_patch(patch: str) -> str:
    if not patch:
        return ""
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Parsing task...")
        )

        try:
            task = json.loads(input_text)
        except json.JSONDecodeError:
            task = {"problem_statement": input_text}

        if not isinstance(task, dict):
            task = {"problem_statement": str(task)}

        instance_id = task.get("instance_id", "?")
        logger.info(f"[{instance_id}] received task")

        user_prompt = _build_user_prompt(task)

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Generating patch for {instance_id}...")
        )

        response_text = await self._call_model(user_prompt)
        patch = _normalize_patch(_extract_patch_from_response(response_text))
        logger.info(f"[{instance_id}] patch length: {len(patch)}")

        artifact = json.dumps({"patch": patch})
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=artifact))],
            name="Result",
        )

    async def _call_model(self, user_prompt: str) -> str:
        # Spread initial fan-out across shards to avoid synchronized burst.
        await asyncio.sleep(random.uniform(0, 5))

        # System prompt with cache_control so subsequent calls in the same
        # shard pay 1/10th the input cost on the system block.
        system_msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        user_msg = {"role": "user", "content": user_prompt}

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                completion = await asyncio.to_thread(
                    litellm.completion,
                    model="anthropic/claude-haiku-4-5-20251001",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    max_tokens=8192,
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if any(s in err for s in ("503", "overloaded", "429", "rate", "limit")):
                    base = min(RETRY_BACKOFF_CAP, 5 * (2 ** min(attempt, 3)))
                    wait = base + random.uniform(0, base * 0.3)  # jitter
                    logger.warning(
                        f"retry {attempt + 1}/{RETRY_MAX_ATTEMPTS} in {wait:.1f}s: {err[:200]}"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"LLM call failed (non-retryable): {err}")
                return ""
        logger.error("LLM call exhausted retries")
        return ""
