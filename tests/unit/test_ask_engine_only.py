"""
Resourcing (Runn today) is a source only on the /ask engine (gub-agent#89, #117).

`config.ASK_CONTRACT` is read at import, so each mode is imported in its own
interpreter. Off (every engine serving the chat bot): the executor is offered
exactly `ALL_TOOLS` and its instruction never mentions resourcing. On: it is
offered the resourcing tools too, their section is appended, and the root is
AskTurn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = """
import asyncio, json
from gub_agent import agent
from gub_agent.tools import ALL_TOOLS
ex = agent.executor_agent
async def text():
    from tests.helpers import invocation_ctx
    from google.adk.agents.readonly_context import ReadonlyContext
    ctx = await invocation_ctx()
    instr, _ = await ex.canonical_instruction(ReadonlyContext(ctx))
    return instr
print(json.dumps({
    "tools": [t.__name__ for t in ex.tools],
    "all": [t.__name__ for t in ALL_TOOLS],
    "root": type(agent.root_agent).__name__,
    "resourcing_in_prompt": "resourcing_allocations" in asyncio.run(text()),
}))
"""


def _probe(flag: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "ASK_CONTRACT"}
    if flag is not None:
        env["ASK_CONTRACT"] = flag
    out = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_the_executor_is_offered_exactly_todays_tools_and_prompt():
    off = _probe(None)
    assert off["tools"] == off["all"]
    assert off["resourcing_in_prompt"] is False
    assert off["root"] == "SequentialAgent"


def test_on_resourcing_is_offered_and_explained_and_the_root_is_the_ask_turn():
    on = _probe("1")
    assert on["tools"] == [*on["all"], "resourcing_allocations", "resourcing_capacity"]
    assert on["resourcing_in_prompt"] is True
    assert on["root"] == "AskTurn"
