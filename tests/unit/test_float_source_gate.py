"""
Float is offered only on an /ask engine that turns it on (config.FLOAT_SOURCE).

Both flags are read at import, so each mode is imported in its own
interpreter. Without FLOAT_SOURCE, an /ask engine is offered exactly what it
was (UB's tools plus Runn's) and its prompt never names Float. With it, the
Float tools follow Runn's and the Float section is appended. FLOAT_SOURCE
alone, on an engine without the /ask contract, changes nothing.
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
    "float_in_prompt": "float_allocations" in asyncio.run(text()),
}))
"""


def _probe(ask: str | None, float_source: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ASK_CONTRACT", "FLOAT_SOURCE")}
    if ask is not None:
        env["ASK_CONTRACT"] = ask
    if float_source is not None:
        env["FLOAT_SOURCE"] = float_source
    out = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_an_ask_engine_without_float_is_unchanged():
    probe = _probe("1", None)
    assert probe["tools"] == [*probe["all"], "runn_allocations", "runn_capacity"]
    assert probe["float_in_prompt"] is False


def test_an_ask_engine_with_float_is_offered_and_told_about_it():
    probe = _probe("1", "1")
    assert probe["tools"] == [
        *probe["all"],
        "runn_allocations",
        "runn_capacity",
        "float_allocations",
        "float_availability",
        "float_capacity",
    ]
    assert probe["float_in_prompt"] is True


def test_float_alone_does_not_touch_the_chat_engine():
    probe = _probe(None, "1")
    assert probe["tools"] == probe["all"]
    assert probe["float_in_prompt"] is False
