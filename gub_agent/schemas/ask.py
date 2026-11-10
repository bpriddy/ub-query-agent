"""
ask.py — the `/ask` contract (gub-agent#88; UB `docs/ASK.md`).

UB's `/ask` returns one answer blended from UB, Runn and the user's Google
Workspace, with every fact tagged by the source it came from. This module is
the shape the interpreter hands UB:

- `AskResponse` is an `AnswerPayload` plus what only the interpreter knows
  after the formatter is done: each fact's source and reference, the UB
  records the answer is about, a status per source, and the conversation id.
  None of it is model-written. The formatter keeps writing `AnswerPayload`
  exactly as before (its `output_schema` is unchanged); `AskResponse` is
  assembled from that payload and the turn's evidence index in code
  (`agents/ask.py`), so a field here can never cost a formatter retry.
- `kind="error"` is new: the real reason the question could not be answered
  (no access, not found, a source down), naming the record. It is distinct from
  `abstain`, which means "I looked and found no answer". The owner's call is
  no obfuscation for now.
- `ProgressEvent` is one line of what the turn is doing, streamed before the
  answer. It carries no model reasoning, tool arguments or document contents:
  its text is written by code from fixed phrases, never by the model.

Field names are snake_case like `AnswerPayload`'s. The public `/ask` shape is
UB's to publish (its OpenAPI document); UB maps this one onto it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .answer import AnswerPayload, Fact

Source = Literal["ub", "runn", "workspace"]
SOURCES: tuple[Source, ...] = ("ub", "runn", "workspace")

SourceStatus = Literal["ok", "error", "timeout", "skipped"]

# A progress line is one short sentence; this keeps a code path that formats
# something unexpected from streaming a paragraph.
PROGRESS_MAX_CHARS = 160


class Ref(BaseModel):
    """A Workspace item a fact was read from: a document, a mail or Chat
    thread, an event. `id` is Google's; the title is for display."""

    type: str
    id: str
    title: str | None = None


class AskFact(Fact):
    """A cited fact with its source. UB facts name their record
    (`entity_type` + `entity_id`); Workspace facts name the item they came
    from (`ref`)."""

    source: Source = "ub"
    entity_type: str | None = None
    ref: Ref | None = None

    @model_validator(mode="after")
    def _workspace_has_ref(self) -> AskFact:
        if self.source == "workspace" and self.ref is None:
            raise ValueError("a workspace fact needs a ref (the item it was read from)")
        return self


class RecordRef(BaseModel):
    """A UB record the answer is about, so a caller can link or act on it."""

    type: str
    id: str
    name: str | None = None


class SourceError(BaseModel):
    code: str
    message: str


class ConflictsWith(BaseModel):
    source: Source
    field: str | None = None


class DroppedClaim(BaseModel):
    """A Workspace claim left out of the answer because UB says otherwise.
    Reported so the caller can tell a document is out of date."""

    value: str
    ref: Ref | None = None
    conflicts_with: ConflictsWith


class SourceReport(BaseModel):
    """One source's outcome. `skipped` means the question did not need it, so
    a caller can tell "not asked" from "failed"."""

    status: SourceStatus
    error: SourceError | None = None
    dropped: list[DroppedClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _error_iff_status_error(self) -> SourceReport:
        if (self.status == "error") != (self.error is not None):
            raise ValueError('a source carries an error exactly when its status is "error"')
        return self


class AskError(BaseModel):
    """Why a question could not be answered, naming the record when there is one."""

    code: str
    message: str
    record: RecordRef | None = None


class AskResponse(AnswerPayload):
    """THE `/ask` answer: the formatter's payload, plus sources, records and
    the conversation, assembled in code."""

    kind: Literal["answer", "abstain", "clarify", "error"] = "answer"  # type: ignore[assignment]
    facts: list[AskFact] = Field(default_factory=list)  # type: ignore[assignment]
    records: list[RecordRef] = Field(default_factory=list)
    sources: dict[Source, SourceReport]
    error: AskError | None = None
    conversation_id: str | None = None

    @model_validator(mode="after")
    def _ask_contract(self) -> AskResponse:
        missing = [s for s in SOURCES if s not in self.sources]
        if missing:
            raise ValueError(f"sources must report every source; missing: {', '.join(missing)}")
        for source, report in self.sources.items():
            if report.dropped and source != "workspace":
                raise ValueError("only workspace claims are dropped (UB wins conflicts)")
        if (self.kind == "error") != (self.error is not None):
            raise ValueError('an error is carried exactly when kind is "error"')
        if self.kind == "error" and (self.blocks or self.citations):
            raise ValueError('kind="error" carries no blocks or citations, only the error')
        return self


ProgressStep = Literal["understand", "lookup", "source", "blend"]


class ProgressEvent(BaseModel):
    """One streamed line: understanding the question, a lookup, a source's
    outcome, or the blend."""

    step: ProgressStep
    text: str = Field(min_length=1, max_length=PROGRESS_MAX_CHARS)
    source: Source | None = None
    status: SourceStatus | None = None

    @model_validator(mode="after")
    def _source_step_names_its_source(self) -> ProgressEvent:
        if (self.step == "source") != (self.source is not None and self.status is not None):
            raise ValueError('a "source" step names its source and status, and only it does')
        return self
