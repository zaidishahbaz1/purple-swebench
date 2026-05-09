"""Local smoke test for the SWE-bench Pro agent.

Bypasses the A2A server entirely — directly imports the Agent class,
constructs a fake message + fake TaskUpdater, and runs the full flow.
Watch stderr for what's actually happening.

Usage:
    cd purple-swebench
    export OPENAI_API_KEY=sk-...
    uv run scripts/local_smoke.py

WARNING: this will:
- pull a SWE-bench Pro instance docker image (~1-2GB) from Docker Hub
- spin up a sidecar container with full repo
- make real OpenAI API calls (gpt-4o + gpt-4o-mini), costs ~$1-3 for
  one task
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

# Make the agent importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import litellm
from a2a.types import Message, Part, Role, TextPart


# ----- Fake task — pick something small & well-known --------------------------
# qutebrowser instances tend to be smallish and quick to pull. Adjust if needed.
TASK = {
    "instance_id": "instance_qutebrowser__qutebrowser-c580ebf0801e5a3ecabc54f327498bb753c6d5f2-v2ef375ac784985212b1805e1d0431dc8f1b3c171",
    "problem_statement": (
        "qutebrowser crashes on startup if a non-ASCII character appears "
        "in the user's home directory path. Fix the encoding handling in "
        "the path-resolution code so the browser starts correctly."
    ),
    "docker_image": "jefzda/sweap-images:qutebrowser.qutebrowser-qutebrowser__qutebrowser-c580ebf0801e5a3ecabc54f327498bb753c6d5f2-v2ef375ac784985212b1805e1d0431dc8f1b3c",
    "base_commit": "c580ebf0801e5a3ecabc54f327498bb753c6d5f2",
    "repo": "qutebrowser/qutebrowser",
    "hints": "",
}


class FakeTaskUpdater:
    """Mimics enough of a2a TaskUpdater that the agent can run."""

    def __init__(self):
        self.statuses = []
        self.artifacts = []

    async def update_status(self, state, message):
        text = ""
        if message and getattr(message, "parts", None):
            for p in message.parts:
                if hasattr(p.root, "text"):
                    text = p.root.text
                    break
        print(f"[STATUS] {state} {text}", file=sys.stderr, flush=True)
        self.statuses.append((state, text))

    async def add_artifact(self, *, parts, name):
        text = parts[0].root.text if parts else ""
        print(f"\n[ARTIFACT name={name}] {text[:500]}", file=sys.stderr, flush=True)
        self.artifacts.append((name, text))


def make_message(task: dict) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(text=json.dumps(task)))],
        message_id=uuid.uuid4().hex,
        context_id="ctx-smoke",
    )


async def main() -> None:
    # Verify env
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Verify docker
    import subprocess
    p = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=10)
    if p.returncode != 0:
        print(f"ERROR: docker not reachable: {p.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[smoke] docker server version: {p.stdout.strip()}", file=sys.stderr)

    # Late import so logging init runs
    from agent import Agent

    print(f"[smoke] starting task: {TASK['instance_id'][:60]}", file=sys.stderr)
    agent = Agent()
    updater = FakeTaskUpdater()
    msg = make_message(TASK)
    await agent.run(msg, updater)

    print("\n[smoke] === FINAL ARTIFACTS ===", file=sys.stderr)
    for name, text in updater.artifacts:
        try:
            j = json.loads(text)
            patch = j.get("patch", "")
            print(f"  artifact name={name} patch_len={len(patch)}", file=sys.stderr)
            if patch:
                preview = patch[:800]
                print(f"  patch preview:\n{preview}", file=sys.stderr)
        except Exception:
            print(f"  artifact name={name} text={text[:200]}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
