"""
ask.py — the `/ask` turn (gub-agent#88; UB `docs/ASK.md`).

`AskTurn` wraps the interpreter's root and turns one run into what UB's
`/ask` streams: progress lines while the turn works, then one `AskResponse`.
It is the root only on an engine deployed with `ASK_CONTRACT` on
(`config.py`); every other engine serves the root exactly as before, so the
chat bot's path does not change.

**Wire.** Every event of the wrapped root is yielded unchanged, because the
runner stores state and history from them. A caller reads only the two kinds
this agent adds, both authored `ask`:

- progress: `custom_metadata={"progress": ProgressEvent}`
- the answer, last: `custom_metadata={"ask_answer": AskResponse}`

Both are `partial=True`: streamed to the caller, never appended to the session
(ADK appends only non-partial events). The conversation's own history keeps the
format gate's payload, as today, so a follow-up turn reads what it always read.

**What is assembled here, in code.** A fact's source comes from the tool that
produced its evidence row (the evidence id's prefix), and a record's type from
the same tool. Runn's tool is in `TOOL_SOURCE` (gub-agent#89).

**Workspace** (gub-agent#90) is not a tool: one Gemini Enterprise call runs
beside the root from the start of the turn (`agents/workspace.py`), as the
person, and its grounded passages join the answer as `workspace` facts with a
`ref` each. The prose stays the root's until the blend (#91). Not configured
on this engine: `skipped`. No token from the service: `error`,
WORKSPACE_NOT_CONNECTED, with the connect page's link. Not answered by the
soft deadline: `timeout`, and the call is cancelled.

**A source's status** comes from its tools' responses: `ok` when any call
succeeded, `error` when every call failed, with the code UB gave (a Runn
`RUNN_NO_GRANT` is a failed source, not a failed turn).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from .. import config
from ..schemas.answer import AnswerPayload
from ..schemas.ask import (
    SOURCES,
    AskError,
    AskFact,
    AskResponse,
    ProgressEvent,
    ProgressStep,
    RecordRef,
    Source,
    SourceError,
    SourceReport,
    SourceStatus,
)
from . import workspace
from .evidence_index import evidence_index
from .formatter import ANSWER_STATE_KEY

logger = logging.getLogger(__name__)

ASK_AUTHOR = "ask"
PROGRESS_KEY = "progress"
ASK_ANSWER_KEY = "ask_answer"

#: The source behind each tool that does not read UB. Every other tool reads UB.
TOOL_SOURCE: dict[str, Source] = {"runn_allocations": "runn"}

#: The UB record type each tool's rows are. `find` and `org_query` span types,
#: so their rows carry none.
TOOL_ENTITY: dict[str, str] = {
    "list_accounts": "account",
    "get_account_overview": "account",
    "get_campaign": "campaign",
    "get_piece": "piece",
    "list_ideas": "idea",
    "get_idea": "idea",
    "search_staff": "staff",
    "get_staff_profile": "staff",
    "find_staff_for_resourcing": "staff",
    "find_files": "file",
}

#: What a lookup progress line says. Fixed phrases, never the call's arguments.
TOOL_PHRASES: dict[str, str] = {
    "find": "Finding what the question names",
    "org_query": "Querying UB's records",
    "search_staff": "Searching staff",
    "get_staff_profile": "Reading a staff profile",
    "find_staff_for_resourcing": "Checking who is available",
    "list_accounts": "Listing accounts",
    "get_account_overview": "Reading an account",
    "get_campaign": "Reading a campaign",
    "get_piece": "Reading a piece",
    "list_ideas": "Listing ideas",
    "get_idea": "Reading an idea",
    "find_files": "Searching file names",
    "runn_allocations": "Checking staffing in Runn",
}

SOURCE_NAMES: dict[Source, str] = {"ub": "UB", "runn": "Runn", "workspace": "Workspace"}


def source_of(tool: str) -> Source:
    return TOOL_SOURCE.get(tool, "ub")


def progress_event(
    ctx: InvocationContext,
    step: ProgressStep,
    text: str,
    *,
    source: Source | None = None,
    status: SourceStatus | None = None,
) -> Event:
    line = ProgressEvent(step=step, text=text, source=source, status=status)
    return Event(
        invocation_id=ctx.invocation_id,
        author=ASK_AUTHOR,
        partial=True,
        custom_metadata={PROGRESS_KEY: line.model_dump(exclude_none=True)},
    )


def answer_event(ctx: InvocationContext, response: AskResponse) -> Event:
    return Event(
        invocation_id=ctx.invocation_id,
        author=ASK_AUTHOR,
        partial=True,
        custom_metadata={ASK_ANSWER_KEY: response.model_dump(exclude_none=True)},
    )


def _tool_of(evidence_id: str, index: dict[str, dict[str, Any]]) -> str:
    entry = index.get(evidence_id)
    if entry and isinstance(entry.get("tool"), str):
        return entry["tool"]
    return evidence_id.split(":", 1)[0]


def _records(facts: list[AskFact], index: dict[str, dict[str, Any]]) -> list[RecordRef]:
    """The UB records the cited facts are about, once each, in citation order.
    A name is attached when the turn's evidence carried one."""
    seen: set[tuple[str, str]] = set()
    records: list[RecordRef] = []
    for fact in facts:
        if fact.source != "ub" or not fact.entity_type or not fact.entity_id:
            continue
        key = (fact.entity_type, fact.entity_id)
        if key in seen:
            continue
        seen.add(key)
        tool = _tool_of(fact.evidence_id, index)
        name = (index.get(f"{tool}:{fact.entity_id}:name") or {}).get("value")
        records.append(
            RecordRef(
                type=fact.entity_type,
                id=fact.entity_id,
                name=name if isinstance(name, str) else None,
            )
        )
    return records


def _counts_as_failure(source: Source, response: dict[str, Any]) -> bool:
    """A tool's error dict that means its source failed. For UB, a 403 or 404
    is an answer about a record (the answer says so), not UB being down; for
    any other source, every error means the source could not be read."""
    if response.get("error") is not True:
        return False
    if source != "ub":
        return True
    status = response.get("status")
    return not isinstance(status, int) or status == 0 or status >= 500


class SourceLog:
    """What each source did this turn: the tools called, in order, and how
    their calls ended. A source is `ok` when any call succeeded, `error` when
    every call failed (with the last failure's code), `skipped` when nothing
    called it."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self._ok: set[Source] = set()
        self._errors: dict[Source, SourceError] = {}
        self._reported: dict[Source, SourceReport] = {}

    def called(self, tool: str) -> bool:
        """Note a call; True the first time this tool is seen."""
        if tool in self.tools:
            return False
        self.tools.append(tool)
        return True

    def answered(self, tool: str, response: Any) -> None:
        source = source_of(tool)
        if isinstance(response, dict) and _counts_as_failure(source, response):
            status = response.get("status")
            self._errors[source] = SourceError(
                code=str(response.get("code") or f"HTTP_{status}"),
                message=str(response.get("message") or "The source could not be read."),
            )
        else:
            self._ok.add(source)

    def report(self, source: Source, report: SourceReport) -> None:
        """A source's outcome decided outside its tools (Workspace's call)."""
        self._reported[source] = report

    def reports(self) -> dict[Source, SourceReport]:
        used = {source_of(tool) for tool in self.tools}
        out: dict[Source, SourceReport] = {}
        for s in SOURCES:
            if s in self._reported:
                out[s] = self._reported[s]
            elif s in self._ok:
                out[s] = SourceReport(status="ok")
            elif s in self._errors:
                out[s] = SourceReport(status="error", error=self._errors[s])
            else:
                out[s] = SourceReport(status="ok" if s in used else "skipped")
        return out


def build_response(
    payload: AnswerPayload,
    index: dict[str, dict[str, Any]],
    log: SourceLog,
    conversation_id: str | None,
    workspace_facts: list[AskFact] | None = None,
) -> AskResponse:
    """The `/ask` answer from the formatter's payload, the turn's evidence and
    Workspace's passages. A clarifying question carries no Workspace facts:
    it is asked before anything is answered."""
    facts = []
    for fact in payload.facts:
        tool = _tool_of(fact.evidence_id, index)
        facts.append(
            AskFact(
                **fact.model_dump(),
                source=source_of(tool),
                entity_type=TOOL_ENTITY.get(tool) if fact.entity_id else None,
            )
        )
    extra = list(workspace_facts or []) if payload.kind != "clarify" else []
    return AskResponse(
        **payload.model_dump(exclude={"facts", "citations"}),
        citations=payload.citations + [f.evidence_id for f in extra],
        facts=facts + extra,
        records=_records(facts, index),
        sources=log.reports(),
        conversation_id=conversation_id,
    )


def error_response(
    code: str, message: str, log: SourceLog, conversation_id: str | None
) -> AskResponse:
    return AskResponse(
        kind="error",
        headline="The question could not be answered",
        error=AskError(code=code, message=message),
        sources=log.reports(),
        conversation_id=conversation_id,
    )


def _function_calls(event: Event) -> list[str]:
    if not event.content or not event.content.parts:
        return []
    return [
        p.function_call.name
        for p in event.content.parts
        if p.function_call and p.function_call.name
    ]


def _function_responses(event: Event) -> list[tuple[str, Any]]:
    if not event.content or not event.content.parts:
        return []
    return [
        (p.function_response.name, p.function_response.response)
        for p in event.content.parts
        if p.function_response and p.function_response.name
    ]


def _question(ctx: InvocationContext) -> str:
    content = ctx.user_content
    if not content or not content.parts:
        return ""
    return " ".join(p.text for p in content.parts if p.text).strip()


class _WorkspaceCall:
    """The turn's one Gemini Enterprise call, started beside the root."""

    def __init__(self, ctx: InvocationContext, log: SourceLog) -> None:
        self.log = log
        self.started = time.monotonic()
        self.task: asyncio.Task | None = None
        self.facts: list[AskFact] = []
        self.items = 0
        self.session_delta: dict[str, Any] = {}
        if not config.WORKSPACE_ENGINE:
            return  # not a source on this engine: skipped
        token = ctx.session.state.get(workspace.TOKEN_STATE_KEY)
        if not isinstance(token, str) or not token:
            log.report(
                "workspace",
                SourceReport(
                    status="error", error=workspace.not_connected(config.WORKSPACE_CONNECT_URL)
                ),
            )
            return
        self.session = ctx.session.state.get(workspace.SESSION_STATE_KEY)
        self.task = asyncio.create_task(
            workspace.stream_assist(
                engine=config.WORKSPACE_ENGINE,
                data_store_ids=config.WORKSPACE_DATA_STORE_IDS,
                token=token,
                query=_question(ctx),
                session=self.session if isinstance(self.session, str) else None,
            )
        )

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def settle(self) -> None:
        """Wait out what is left of the soft deadline, then report."""
        if self.task is None:
            return
        remaining = config.WORKSPACE_SOFT_DEADLINE_S - (time.monotonic() - self.started)
        done, _ = await asyncio.wait({self.task}, timeout=max(0.0, remaining))
        if not done:
            self.cancel()
            self.log.report("workspace", SourceReport(status="timeout"))
            return
        try:
            result, rejected = self.task.result()
        except workspace.StreamAssistError as err:
            logger.warning("ask workspace: status=%s reason=%s", err.status, err.reason)
            if err.reason == "TIMEOUT":
                self.log.report("workspace", SourceReport(status="timeout"))
            else:
                self.log.report(
                    "workspace",
                    SourceReport(
                        status="error", error=workspace.source_error(err, config.WORKSPACE_APP_URL)
                    ),
                )
            return
        except Exception:  # noqa: BLE001 — a Workspace failure never fails the answer
            logger.exception("ask workspace failed")
            self.log.report(
                "workspace",
                SourceReport(
                    status="error",
                    error=SourceError(
                        code="WORKSPACE_UNAVAILABLE", message="Gemini Enterprise could not be read."
                    ),
                ),
            )
            return
        if result.unauthorized_connectors:
            logger.info(
                "ask workspace: connectors not authorised: %d", len(result.unauthorized_connectors)
            )
        self.facts = workspace.facts_of(result)
        self.items = len({f.ref.id for f in self.facts if f.ref})
        self.log.report("workspace", SourceReport(status="ok"))
        if result.session and result.session != self.session:
            self.session_delta = {workspace.SESSION_STATE_KEY: result.session}
        elif rejected:
            self.session_delta = {workspace.SESSION_STATE_KEY: None}


class AskTurn(BaseAgent):
    """The interpreter's root, run as one `/ask` turn (see module docstring)."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        [root] = self.sub_agents
        conversation_id = ctx.session.id
        log = SourceLog()
        payload: dict[str, Any] | None = None

        yield progress_event(ctx, "understand", "Reading the question")
        ws = _WorkspaceCall(ctx, log)
        if ws.task is not None:
            yield progress_event(ctx, "lookup", "Searching your Workspace")
        try:
            async for event in root.run_async(ctx):
                yield event
                for tool, response in _function_responses(event):
                    log.answered(tool, response)
                for tool in _function_calls(event):
                    if log.called(tool):
                        yield progress_event(
                            ctx, "lookup", TOOL_PHRASES.get(tool, "Looking something up")
                        )
                delta = event.actions.state_delta if event.actions else None
                if delta and ANSWER_STATE_KEY in delta:
                    payload = delta[ANSWER_STATE_KEY]
        except Exception as err:  # noqa: BLE001 — the caller always gets an answer event
            ws.cancel()
            logger.exception("ask turn failed")
            yield answer_event(
                ctx,
                error_response("INTERNAL", f"The interpreter failed: {err}", log, conversation_id),
            )
            return

        if payload is None:
            ws.cancel()
            yield answer_event(
                ctx,
                error_response(
                    "NO_ANSWER",
                    "The interpreter finished without an answer.",
                    log,
                    conversation_id,
                ),
            )
            return

        await ws.settle()
        if ws.session_delta:
            # Not partial, so the runner saves it: the next turn of this
            # conversation continues the same Gemini Enterprise session. No
            # content, so it is no part of the conversation's history.
            yield Event(
                invocation_id=ctx.invocation_id,
                author=ASK_AUTHOR,
                actions=EventActions(state_delta=ws.session_delta),
            )

        response = build_response(
            AnswerPayload.model_validate(payload),
            evidence_index(ctx.invocation_id),
            log,
            conversation_id,
            workspace_facts=ws.facts,
        )
        for source, report in response.sources.items():
            if report.status == "skipped":
                continue
            name = SOURCE_NAMES[source]
            found = {"ub": len(response.records), "workspace": ws.items}.get(source, 0)
            noun = "item" if source == "workspace" else "record"
            if report.error is not None:
                text = f"Couldn't read {name} ({report.error.code})"
            elif report.status == "timeout":
                text = f"{name} didn't answer in time"
            elif found:
                text = f"Found {found} {noun}{'s' if found != 1 else ''} in {name}"
            else:
                text = f"Checked {name}"
            yield progress_event(ctx, "source", text, source=source, status=report.status)
        yield answer_event(ctx, response)
