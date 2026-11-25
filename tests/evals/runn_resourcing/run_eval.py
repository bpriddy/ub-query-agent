"""
run_eval.py — run the Runn resourcing questions through the real `/ask` agent.

    GOOGLE_CLOUD_PROJECT=os-test-491819 \\
        .venv/bin/python -m tests.evals.runn_resourcing.run_eval --out /tmp/runn-eval

`--source float` runs the Float subset (questions.FLOAT_SUBSET) against the
same agency in Float, with FLOAT_SOURCE on and the Runn relay not configured.

What it does, per question (questions.py):
  * builds the real root agent with ASK_CONTRACT=1 (the `/ask` engine's tree:
    router, dispatcher, executor with the resourcing tools, format gate, critic)
    and runs it on an ADK InMemoryRunner, one session per chain, so a
    follow-up reads the turn before it;
  * points every UB tool at a localhost fake UB (fake_ub.py) serving the
    fixtures through the relay's routes, and nothing more;
  * pins "today" to world.TODAY in the agent's instructions;
  * uses the real Gemini model on Vertex with application-default
    credentials (GOOGLE_CLOUD_PROJECT, location global, as config.py pins);
  * records the tool calls, the relay calls, and the `/ask` answer, then
    grades the answer against the ground truth: names found / forbidden names
    put forward (in code), and a verdict + failure type from a judge model
    that is given the truth, the tool trace and the answer.

Costs real tokens (about 37 agent turns plus 37 judge calls). Never runs in
CI: the hermetic tests (tests/evals/test_runn_resourcing_eval.py) cover the
fixtures, the ground truth and the fake relay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Which planner the fake UB has: Runn (the full question set), or Float with
# Runn not configured (the Float subset). Read before gub_agent is imported,
# because FLOAT_SOURCE decides at import whether the Float tools are offered.
def _source(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--source=float" or (arg == "--source" and argv[i + 1 : i + 2] == ["float"]):
            return "float"
    return "runn"


SOURCE = _source(sys.argv)

# Before gub_agent is imported: config.py reads these at import.
os.environ.setdefault("GUB_BASE_URL", "http://127.0.0.1:9")  # replaced once the fake is up
os.environ["ASK_CONTRACT"] = "1"
os.environ["FLOAT_SOURCE"] = "1" if SOURCE == "float" else "0"
os.environ["GUB_SERVICE_JWT"] = "eval"  # a label, not a credential: the fake UB accepts any
os.environ["SANDBOX_ENABLED"] = "0"
os.environ["FILE_SEARCH_ENABLED"] = "0"
os.environ["WORKSPACE_ENGINE"] = ""
os.environ["EMIT_THINKING"] = "0"
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "os-test-491819")

from . import questions as questions_module  # noqa: E402
from . import world  # noqa: E402
from .fake_ub import FakeUb  # noqa: E402
from .truth import Truth  # noqa: E402

JUDGE_MODEL = os.environ.get("RUNN_EVAL_JUDGE_MODEL", "gemini-2.5-pro")

FAILURE_TYPES = [
    "none",
    "data_not_exposed",  # the data exists in Runn but the relay/tool does not return it
    "cannot_enumerate",  # the tool needs one person or project; the question is about many
    "office_join",  # office could not be applied (UB office x Runn person)
    "role_or_skill",  # role/skill could not be applied or was conflated
    "time_off_ignored",  # leave or public holidays not taken into account
    "tentative_ignored",  # tentative vs confirmed not distinguished
    "part_time_ignored",  # contract hours / rostered days not taken into account
    "identity_mismatch",  # person missing from Runn or a different email; wrongly concluded
    "wrong_dates",  # relative dates resolved wrongly
    "arithmetic",  # hours / percentages computed wrongly
    "hallucination",  # stated something no tool returned
    "no_clarification",  # should have asked which one (or answered for each)
    "gave_up",  # abstained or said it cannot answer when the tools could
    "other",
]


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: ARG003 — datetime's signature
        return datetime(world.TODAY.year, world.TODAY.month, world.TODAY.day, 15, 0, tzinfo=UTC)


def _answer_text(answer: dict | None) -> str:
    if not answer:
        return ""
    out: list[str] = [answer.get("headline") or ""]

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k not in ("type", "id", "kind", "style"):
                    walk(v)

    walk(answer.get("blocks") or [])
    walk(answer.get("candidates") or [])
    return "\n".join(x for x in out if x)


def _mentions(text: str, name: str) -> bool:
    low = text.lower()
    if name.lower() in low:
        return True
    if " - " in name:  # a project: its distinctive half is enough
        return name.split(" - ", 1)[1].lower() in low
    return False


def _grade_names(q: dict, text: str) -> dict:
    t = q["truth"]
    core_hit = [n for n in t["core"] if _mentions(text, n)]
    return {
        "core_found": f"{len(core_hit)}/{len(t['core'])}",
        "core_missing": [n for n in t["core"] if n not in core_hit],
        "forbidden_mentioned": [n for n in t["forbidden"] if _mentions(text, n)],
        "other_named": [],
    }


def _short(value: Any, limit: int = 600) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


async def _judge(client: Any, q: dict, prior: list[dict], record: dict) -> dict:
    from google.genai import types

    truth = q["truth"]
    truth_text = (
        truth["expected"]
        if "expected" in truth
        else (
            f"MUST name: {truth['core']}. MAY also name: {truth['acceptable']}. MUST NOT put "
            f"forward as an answer (mentioning them as unavailable is fine): {truth['forbidden']}. "
            f"Context: {truth['note']}"
        )
    )
    planner = "Float" if SOURCE == "float" else "Runn"
    prompt = f"""You grade an AI assistant's answer to a resource manager's question about
staffing in {planner} (the resource planner). Today is {world.TODAY.isoformat()} (a Wednesday).

Earlier turns in this conversation: {json.dumps([p["text"] for p in prior]) or "none"}
Question: {q["text"]}

GROUND TRUTH (computed from the full {planner} data): {truth_text}

The assistant's tool calls (name, arguments, start of the result):
{json.dumps(record["tool_calls"], indent=0)[:6000]}

The HTTP calls those tools made to UB (UB's own records under /org, the planner under
/relay/resourcing (Runn) or /relay/float; the fast path calls UB in code, so this list can
be longer):
{json.dumps(record["relay_calls"], indent=0)[:3000]}

The assistant's answer:
<<<
{record["answer_text"]}
>>>

Grade it:
- pass: correct and complete enough that a resource manager could act on it (small omissions
  of acceptable names are fine; honest caveats are fine).
- partial: some of it is right, but it misses core people/facts or adds a wrong one.
- fail: wrong, misleading, made up, or no real answer when one was possible. An honest
  "I can't tell from the data I have" is a fail too (the data exists in {planner}), but give it
  the failure type that says WHY it could not answer.
failure_type: the main reason it is not a pass ("none" for a pass), one of {FAILURE_TYPES}.
Use data_not_exposed / cannot_enumerate when the tools it had could not return what was
needed (look at the tool calls), and the specific reasoning type when the data was there.
reason: one short sentence."""
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "partial", "fail"]},
            "failure_type": {"type": "string", "enum": FAILURE_TYPES},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "failure_type", "reason"],
    }
    for attempt in range(3):
        try:
            resp = await client.aio.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                ),
            )
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001 — a judge failure is recorded, not fatal
            if attempt == 2:
                return {"verdict": "error", "failure_type": "other", "reason": repr(exc)[:300]}
            await asyncio.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


async def _run_turn(runner: Any, session_id: str, text: str) -> dict:
    from google.genai import types as genai_types

    from gub_agent.agents.ask import ASK_ANSWER_KEY

    calls: list[dict] = []
    by_id: dict[str, dict] = {}
    answer = None
    started = time.monotonic()
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
    error = None
    try:
        async for event in runner.run_async(
            user_id="eval", session_id=session_id, new_message=message
        ):
            # A partial event streams a call before its arguments are complete;
            # the final event carries it again, whole.
            for fc in [] if event.partial else event.get_function_calls() or []:
                entry = {"tool": fc.name, "args": dict(fc.args or {}), "result": None}
                calls.append(entry)
                if fc.id:
                    by_id[fc.id] = entry
            for fr in event.get_function_responses() or []:
                entry = by_id.get(fr.id or "")
                if entry is not None:
                    entry["result"] = _short(fr.response)
            meta = event.custom_metadata or {}
            if ASK_ANSWER_KEY in meta:
                answer = meta[ASK_ANSWER_KEY]
    except Exception as exc:  # noqa: BLE001 — recorded as the turn's outcome
        error = repr(exc)[:500]
    return {
        "tool_calls": calls,
        "answer": answer,
        "answer_text": _answer_text(answer),
        "error": error,
        "seconds": round(time.monotonic() - started, 1),
    }


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="runn-eval-results", help="directory for results")
    parser.add_argument("--only", nargs="*", help="question ids (their whole chains run)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--source",
        choices=["runn", "float"],
        default="runn",
        help="runn: every question, Runn configured; float: the Float subset, Runn not",
    )
    args = parser.parse_args(argv)

    data = world.load()
    truth = Truth(data)
    qs = (questions_module.build_float if SOURCE == "float" else questions_module.build)(truth)
    if args.only:
        chains = {q["chain"] for q in qs if q["id"] in args.only}
        qs = [q for q in qs if q["chain"] in chains]

    fake = FakeUb(data, sources=(SOURCE,))
    await fake.start()

    from google import genai
    from google.adk.runners import InMemoryRunner

    from gub_agent import instruction_utils
    from gub_agent.agent import root_agent
    from gub_agent.tools import _client

    instruction_utils.datetime = _Frozen
    _client.GUB_BASE_URL = fake.url

    runner = InMemoryRunner(agent=root_agent, app_name="gub_agent")
    judge = genai.Client(
        vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global"
    )
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, dict] = {}

    chains: dict[str, list[dict]] = {}
    for q in qs:
        chains.setdefault(q["chain"], []).append(q)

    async def run_chain(chain: str, items: list[dict]) -> None:
        async with sem:
            token = f"eval-{chain}"
            session = await runner.session_service.create_session(
                app_name="gub_agent", user_id="eval", state={"gub_jwt": token}
            )
            prior: list[dict] = []
            for q in items:
                seen = len(fake.log[token])
                record = await _run_turn(runner, session.id, q["text"])
                record["relay_calls"] = [
                    f"{c['method']} {c['path']}"
                    + (
                        f"?{'&'.join(f'{k}={v}' for k, v in c['params'].items())}"
                        if c["params"]
                        else ""
                    )
                    + (f" {_short(c['body'], 300)}" if c["body"] else "")
                    + f" -> {c['status']}"
                    for c in fake.log[token][seen:]
                ]
                if q["kind"] == "names":
                    record["names"] = _grade_names(q, record["answer_text"])
                record["judge"] = (
                    {"verdict": "skipped", "failure_type": "other", "reason": ""}
                    if args.no_judge
                    else await _judge(judge, q, prior, record)
                )
                results[q["id"]] = {**q, **record}
                prior.append(q)
                v = record["judge"]
                print(
                    f"{v['verdict']:8} {q['id']:32} {v['failure_type']:18} "
                    f"{len(record['tool_calls'])} calls {record['seconds']}s",
                    flush=True,
                )

    try:
        await asyncio.gather(*(run_chain(c, items) for c, items in chains.items()))
    finally:
        await _client.aclose_client()
        await fake.stop()

    ordered = [results[q["id"]] for q in qs if q["id"] in results]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(ordered, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(summary(ordered))
    print(summary(ordered))
    return 0


def summary(rows: list[dict]) -> str:
    score = {"pass": 1.0, "partial": 0.5}
    total = sum(score.get(r["judge"]["verdict"], 0.0) for r in rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["judge"]["verdict"]] = counts.get(r["judge"]["verdict"], 0) + 1
    lines = [
        f"# Runn resourcing eval ({SOURCE}) — {world.TODAY.isoformat()} pinned",
        "",
        f"Score: {total:g}/{len(rows)} (pass 1, partial 0.5) — "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        "",
        "| id | today | verdict | failure | tools | reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        tools = ",".join(sorted({c["tool"] for c in r["tool_calls"]})) or "-"
        reason = re.sub(r"\s+", " ", r["judge"]["reason"]).replace("|", "/")
        lines.append(
            f"| {r['id']} | {r['today']} | {r['judge']['verdict']} | "
            f"{r['judge']['failure_type']} | {tools} | {reason} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
