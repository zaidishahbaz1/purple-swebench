"""Docker-only smoke test (no LLM calls).

Verifies start_container + bash exec inside the sidecar works for a real
SWE-bench Pro image. No OpenAI calls, costs $0.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent, _run

IMAGE = "jefzda/sweap-images:qutebrowser.qutebrowser-qutebrowser__qutebrowser-c580ebf0801e5a3ecabc54f327498bb753c6d5f2-v2ef375ac784985212b1805e1d0431dc8f1b3c"


class FakeUpdater:
    async def update_status(self, *a, **kw):
        pass

    async def add_artifact(self, *a, **kw):
        pass


async def main():
    agent = Agent()
    name = "docker-smoke-test"
    _run(["docker", "rm", "-f", name], None, 10)

    ok = await agent._start_container(IMAGE, name, FakeUpdater(), "test")
    print(f"\n=== start_container returned: {ok} ===")

    if ok:
        # Try some real bash commands
        for cmd in ["uname -a", "ls /app | head -10", "cat /app/qutebrowser/version.py 2>&1 | head -20", "git -C /app log --oneline -3"]:
            rc, out, err = agent._exec_input(name, ["sh", "-c", cmd], "", 30)
            print(f"\n$ {cmd}")
            print(f"  rc={rc}")
            print(f"  stdout: {out[:300]!r}")
            print(f"  stderr: {err[:200]!r}")

    _run(["docker", "rm", "-f", name], None, 30)
    print("\n=== done ===")


if __name__ == "__main__":
    asyncio.run(main())
