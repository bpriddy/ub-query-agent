"""
fast_path.py — one deterministic lookup instead of the ReAct loop (blend 04).

A `BaseAgent` with NO model call before the formatter. For a FACT intent with
an unambiguous entity the whole answer is one HTTP request, so this path makes
that request itself, feeds the result into the turn's evidence index, and runs
the format gate — the same gate, the same `AnswerPayload`, the same evidence
ids as the deep path. What it skips is the executor's 1-3 model rounds at
MEDIUM thinking and the critic pass after them: 20-45 s of model turns for
information one call already has.

Four choices worth their comments:

- **The tool functions are reused, not reimplemented.** `get_campaign` and
  friends are ordinary async functions with `tool_context: Any = None`
  (`tools/accounts.py:92`), and `_resolve_gub_jwt` only ever touches
  `tool_context.state` (`tools/_client.py:68-80`). `_ToolShim` supplies that
  one field from `ctx.session.state`, so the session's cached `gub_jwt` is
  reused with no re-exchange and a new HTTP layer never comes into existence.
- **The critic does not run here.** Grounding is the format gate's job (in
  code, blend 03) and information sufficiency is guaranteed by construction: a
  deterministic call for a known entity cannot have made the wrong tool
  choice. There is nothing left for the judge to judge.
- **It never guesses an entity.** Resolution mirrors the bot's own rules
  (`gub-gchat-bot/src/entity/resolve.ts`, blend 02) with the same thresholds;
  anything ambiguous goes to the deep path WITHOUT a clarification card — the
  bot may already have spent this turn's one card, and a second question is
  worse than a slower answer.
- **`similar_to` is structurally out of reach.** The `count_or_rank` builder
  resolves names through `/org/search` and filters by id, so it never emits
  the one operator that must be a call's sole filter
  (`tools/org_query.py` docstring). It cannot violate that rule by
  construction rather than by care.

Failure policy: an empty result, an unassemblable query or an unexpected
exception falls through to the deep path ONCE (the executor may still find an
answer); a 403 or a 404 is answered immediately, because paying 30 s of ReAct
to hear the same 403 helps nobody. The exception is an id the BOT supplied (its
"User selected …" / "Best guess: …" prefix): the fast path did not find that
record, it was handed it, and a refusal about a handed id is no evidence about
what the reader asked, so there the deep path runs too. And an id is used at
all only when the prefix says its type is the one the lookup reads
(`_campaign_id`).
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from .. import config
from ..config import FAST_TIE_BAND
from ..schemas import AnswerPayload
from ..schemas.router import RouterDecision
from ..tenant import label_of
from ..tools.accounts import get_account_overview, get_campaign
from ..tools.discovery import find
from ..tools.org_query import org_query
from ..tools.staff import get_staff_profile, search_staff
from .answers import (
    is_record_id_field,
    no_access_payload,
    not_found_payload,
    payload_event,
    tool_activity_event,
)
from .evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
    set_answer_draft,
)
from .format_gate import format_gate
from .router import decision_from, user_text

logger = logging.getLogger(__name__)

# How much of one tool result the draft quotes. The formatter also receives
# every indexed value under ALLOWED_EVIDENCE, so the draft is orientation, not
# the data itself.
MAX_DRAFT_FIELD_CHARS = 600
MAX_DRAFT_ROWS = 10

# The bot's word for a campaign in its "User selected …" / "Best guess: …"
# prefix (`entity/resolve.ts` CLARIFIABLE_TYPES).
CAMPAIGN_TYPE = "campaign"

# The precheck's two reasons for a campaign intent that carries an id which is
# not (known to be) a campaign's. Apart on purpose: with the precheck on, the
# turn never reaches `_campaign`, so its "is not a campaign entity_type=-" line
# is never written — and the slug is then the only trace of an id the bot's
# prefix did NOT vouch for (the router invented or mangled it, or the chat
# client reworded the prefix), which must stay countable apart from the
# ordinary piece / account / person turns.
TYPED_ID_REASON = "entity_type"  # the prefix says piece / account / staff / …
UNVOUCHED_ID_REASON = "entity_type_unknown"  # no prefix vouched for the id

# ── the fast path's outcome, for the dispatcher ───────────────────────────────
#
# An async generator cannot return a value alongside its events, and inferring
# "did it answer?" from the event stream would couple the dispatcher to the
# gate's internals. Same in-process, invocation-keyed store as
# `round_limiter.py` (flat `state` writes do not survive between calls).

_OUTCOME: OrderedDict[str, str] = OrderedDict()
_MAX_TRACKED = 256


def _set_outcome(invocation_id: str, outcome: str) -> None:
    _OUTCOME[invocation_id] = outcome
    _OUTCOME.move_to_end(invocation_id)
    while len(_OUTCOME) > _MAX_TRACKED:
        _OUTCOME.popitem(last=False)


def outcome(invocation_id: str) -> str:
    """`"answered"` when the fast path emitted a payload for this invocation,
    `"deep"` otherwise (including "it never ran")."""
    return _OUTCOME.get(invocation_id, "deep")


# ── the tool-context shim ─────────────────────────────────────────────────────


class _ToolShim:
    """The one field the tool functions actually use.

    `_resolve_gub_jwt` reads `state["gub_jwt"]` and, on a cold session,
    exchanges the injected OAuth token and writes the result back
    (`tools/_client.py:68-108`). That write goes to the live session-state
    object and is NOT committed as a state_delta — the same limitation the
    whole codebase works around — so a cold session that then falls through to
    the deep path pays one extra exchange. Correct either way, and the warm
    case (the bot's sessions are long-lived) is the common one.

    `invocation_id` is here for `record_evidence`, which reads it off the tool
    context to key the turn's index.
    """

    def __init__(self, ctx: InvocationContext) -> None:
        self.state = ctx.session.state
        self.invocation_id = ctx.invocation_id
        #: The user's previous message in this session, for a follow-up
        #: ("Jane Park, in LA.") whose subject the earlier turn set.
        self.prior_question = _prior_user_text(ctx)


def _prior_user_text(ctx: Any) -> str:
    """The user's message before this invocation's, or ""."""
    session = getattr(ctx, "session", None)
    for event in reversed(getattr(session, "events", None) or []):
        if event.author != "user" or event.invocation_id == ctx.invocation_id:
            continue
        parts = event.content.parts if event.content and event.content.parts else []
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        if text.strip():
            return text
    return ""


def _where(shim: Any) -> tuple[Any, str]:
    """`(inv, tenant)` for a reason line. Read defensively: a log call must
    never fail the turn, and the pure builders are also driven with bare
    stand-ins."""
    return getattr(shim, "invocation_id", None), label_of(getattr(shim, "state", None))


# ── entity resolution ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    """`kind`: "one" (id resolved), "ambiguous", "none". Only "one" carries an
    id — an ambiguous surface is never picked from."""

    kind: str
    entity_id: str | None = None
    name: str | None = None


def resolve_hits(hits: list[dict], surface: str, wanted: tuple[str, ...]) -> Resolution:
    """The bot's disambiguation rule (`entity/resolve.ts:initialDecision`) as a
    pure function over `/org/search` hits, restricted to the types this intent
    can use:

    - exactly one hit whose NAME equals the surface → that one, however many
      others sit beside it;
    - two or more hits with that same exact name → ambiguous (the canonical
      case: three campaigns all called "Silverado");
    - otherwise build the set TIED at the top (within FAST_TIE_BAND) and take
      it if it holds one candidate, ask if it holds several.

    The rule this replaces ended `top - second < margin AND top < exact floor`,
    and the second clause read a high top score as confidence. Since
    `/org/search` began scoring by WORD match, every candidate a query names
    equally well scores 1.0 — so that clause fired hardest on the clearest
    ambiguities and suppressed the question. A tie band cannot invert that way:
    it asks only how many candidates the search could not separate.
    """
    typed = [
        h
        for h in hits
        if isinstance(h, dict) and h.get("type") in wanted and isinstance(h.get("id"), str)
    ]
    if not typed:
        return Resolution("none")
    typed.sort(key=lambda h: float(h.get("similarity") or 0.0), reverse=True)

    exact = [h for h in typed if str(h.get("name", "")).lower() == surface.lower()]
    if len(exact) == 1:
        return Resolution("one", exact[0]["id"], exact[0].get("name"))
    if len(exact) >= 2:
        return Resolution("ambiguous")

    top = float(typed[0].get("similarity") or 0.0)
    tied = [h for h in typed if top - float(h.get("similarity") or 0.0) <= FAST_TIE_BAND]
    if len(tied) == 1:
        return Resolution("one", tied[0]["id"], tied[0].get("name"))
    return Resolution("ambiguous")


async def _resolve(surface: str | None, wanted: tuple[str, ...], shim: _ToolShim) -> Resolution:
    """Resolve a surface through the existing `find` tool (`GET /org/search`,
    `tools/discovery.py:49`) — the same endpoint, the same access scoping the
    deep path uses."""
    if not surface or not surface.strip():
        return Resolution("none")
    response = await find(surface, shim)
    # `GET /org/search` answers with a BARE ARRAY (org.controller.ts: `res.json(
    # results)`), and `gub_get` returns `resp.json()` verbatim — so the happy
    # path here is a list, not a dict. Only the error envelope is a dict. This
    # used to test `isinstance(response, dict)` FIRST and bail, which made every
    # real response resolve to "none" and the unreachable `hits` lookup below it
    # look like the working branch; the unit tests missed it because they stub
    # `{"hits": [...]}`, a shape GUB has never emitted.
    if isinstance(response, dict):
        if response.get("error"):
            return Resolution("none")
        hits = response.get("hits")
        hits = hits if isinstance(hits, list) else []
    elif isinstance(response, list):
        hits = response
    else:
        return Resolution("none")
    return resolve_hits(hits, surface.strip(), wanted)


#: How `org_query_args` turns an account name into an id: `_resolve`'s shape.
Resolver = Callable[[str | None, tuple[str, ...], Any], Awaitable[Resolution]]


# ── the org_query builder (count_or_rank) ─────────────────────────────────────

# Status vocabularies are ENTITY-SPECIFIC and a value from the wrong entity
# silently matches nothing (`tools/org_query.py` docstring) — so a status the
# entity does not know is a deep-path question, not a filter.
_STATUS_VALUES = {
    "campaigns": {"pitch", "awarded", "in_progress", "live", "paused", "ended", "lost"},
    "accounts": {"active", "inactive", "prospect"},
    "staff": {"active", "on_leave", "former"},
    "pieces": set(),
}

# The only date field the builder will filter on, per entity.
_PERIOD_FIELD = {"campaigns": "awardedAt"}

# The only numeric metric it will rank by, per entity.
_METRIC_FIELDS = {"campaigns": {"budget": "budget"}}

# What the sentence asks FOR. Checked against the user's own words because
# `Slots` has no field for it: a count, a ranking and a sum are three different
# answers built from identical slots.
_COUNT_RE = re.compile(
    r"\b(how many|how much|number of|count of|сколько|количество)\b",
    re.I,
)
_RANK_RE = re.compile(
    r"\b(largest|biggest|highest|greatest|top|most expensive|smallest|lowest|cheapest"
    r"|самый|самая|самое|крупнейш\w*|наибольш\w*|наименьш\w*)\b",
    re.I,
)
# Aggregates this builder cannot express, and enumerations it must not answer
# with a number. Both belong to the deep path.
_AGGREGATE_RE = re.compile(
    r"\b(combined|total(?!\s+of\s+\d)|sum|average|mean|median|суммарн\w*|средн\w*|в сумме|итого)\b",
    re.I,
)
_ENUMERATE_RE = re.compile(
    r"\b(which|what are|list|name the|show me|give me the (?:names|list)"
    r"|какие|какой из|перечисл\w*|назови|покажи)\b",
    re.I,
)


def _asked_shape(question: str) -> str:
    """ "count" | "rank" | "other" — what the sentence wants back.

    Order matters. An enumeration or an aggregate is disqualifying even when
    counting words are also present ("list how many we have per office"), so
    those are tested first; a ranking beats a bare count ("which campaign has
    the largest budget" contains neither "how many" nor a plain list).
    """
    if _AGGREGATE_RE.search(question):
        return "other"
    if _RANK_RE.search(question):
        return "rank"
    if _ENUMERATE_RE.search(question):
        return "other"
    if _COUNT_RE.search(question):
        return "count"
    return "other"


# Fields it will group by, per entity — an FK id group carries its `*Name`
# companion in the result rows, so a group-by needs no follow-up query.
_GROUP_FIELDS = {
    "campaigns": {"status": "status", "account": "accountId", "accountid": "accountId"},
    "accounts": {"status": "status", "industry": "industry"},
    "staff": {"office": "officeId", "officeid": "officeId", "status": "status"},
    "pieces": {"campaign": "campaignId", "campaignid": "campaignId"},
}

_YEAR_RE = re.compile(r"^(\d{4})$")
_QUARTER_RE = re.compile(r"^(?:(\d{4})\s*[- ]?\s*q([1-4])|q([1-4])\s*[- ]?\s*(\d{4}))$")
_QUARTER_RANGES = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}


def period_range(text: str) -> list[str] | None:
    """`["YYYY-MM-DD", "YYYY-MM-DD"]` for an explicit year or quarter, else
    None. Deliberately narrow: a relative period ("this year", "recently")
    needs a judgement about what the user meant, which is the executor's job,
    not a regex's."""
    value = text.strip().lower()
    year = _YEAR_RE.match(value)
    if year:
        return [f"{year.group(1)}-01-01", f"{year.group(1)}-12-31"]
    quarter = _QUARTER_RE.match(value)
    if quarter:
        y = quarter.group(1) or quarter.group(4)
        q = int(quarter.group(2) or quarter.group(3))
        start, end = _QUARTER_RANGES[q]
        return [f"{y}-{start}", f"{y}-{end}"]
    return None


async def org_query_args(
    decision: RouterDecision,
    shim: _ToolShim,
    question: str,
    *,
    resolve: Resolver | None = None,
    why: list[str] | None = None,
) -> dict[str, Any] | None:
    """`org_query` kwargs assembled from the router's `slots`, or None → deep
    path.

    Conservative in three layers, because a count that quietly dropped a
    constraint would answer a DIFFERENT question than the one asked — the one
    failure mode worse than being slow:

    1. `slots.complete` must be true — the router's own statement that the
       named fields express the whole sentence (`schemas/router.py`).
    2. `slots.office` is refused outright: an office needs an id this path
       cannot resolve deterministically.
    3. every remaining value must map to a field the NAMED entity actually
       has, with an entity-specific status vocabulary (a status from another
       entity silently matches nothing).

    `resolve` and `why` exist for `declined_before_lookup`, which asks this
    same builder whether it would decline no matter what the account name
    resolves to: `resolve` stands in for `_resolve` (None → `_resolve`, looked
    up at call time), and `why` collects the reason for a None. The lookup
    passes neither, so its call is what it always was.
    """
    resolve = resolve or _resolve

    def declined(reason: str) -> None:
        if why is not None:
            why.append(reason)
        return None

    slots = decision.slots
    if not slots.complete:
        logger.info(
            "fast_path: router says the slots do not cover the question — deep path"
            " (inv=%s) tenant=%s",
            *_where(shim),
        )
        return declined("slots_incomplete")
    if slots.office:
        logger.info(
            "fast_path: office constraint needs an id lookup — deep path (inv=%s) tenant=%s",
            *_where(shim),
        )
        return declined("office")

    entity = slots.entity
    if entity is None:
        return declined("no_entity")

    args: dict[str, Any] = {"entity": entity}
    filters: dict[str, dict[str, Any]] = {}

    if slots.status:
        status = slots.status.strip().lower()
        if status not in _STATUS_VALUES[entity]:
            return declined("status_vocab")
        filters["status"] = {"eq": status}

    if slots.industry:
        if entity != "accounts":
            return declined("industry_entity")
        filters["industry"] = {"eq": slots.industry.strip()}

    if slots.period:
        field_name = _PERIOD_FIELD.get(entity)
        window = period_range(slots.period)
        if field_name is None or window is None:
            return declined("period")
        filters[field_name] = {"between": window}

    if slots.account:
        if entity not in ("campaigns", "accounts"):
            return declined("account_entity")
        # Resolve the name to an id and filter on the id — never `similar_to`,
        # which v1 requires to be a call's SOLE filter.
        resolution = await resolve(slots.account, ("account",), shim)
        if resolution.kind != "one" or resolution.entity_id is None:
            return declined("account_unresolved")
        filters["accountId" if entity == "campaigns" else "id"] = {"eq": resolution.entity_id}

    if filters:
        args["filter"] = filters

    if slots.group_by:
        field_name = _GROUP_FIELDS[entity].get(slots.group_by.strip().lower())
        if field_name is None:
            return declined("group_by")
        args["group_by"] = [field_name]

    if slots.limit is not None:
        if not 1 <= slots.limit <= 100:
            return declined("limit")
        args["limit"] = slots.limit

    # What SHAPE of answer the sentence asked for. `slots` cannot tell these
    # apart — "how many campaigns", "which campaigns" and "the combined budget
    # of the campaigns" all arrive as the same entity/status/metric fields — and
    # answering one with another is how this builder produced four confidently
    # wrong answers out of eight in the 2026-09-10 eval: a count for "which
    # accounts do we have", a ranking for "the combined budget".
    shape = _asked_shape(question)
    if shape == "rank":
        if not slots.metric:
            return declined("rank_no_metric")
        field_name = _METRIC_FIELDS.get(entity, {}).get(slots.metric.strip().lower())
        if field_name is None:
            return declined("rank_metric")
        # A ranking: the DB sorts, we take the top rows. NULLs must be excluded
        # or they sort first and the "largest budget" is a row with no budget —
        # which is exactly what `fact-15` returned.
        filters[field_name] = {"is_null": False}
        args["filter"] = filters
        args["sort"] = [{"field": field_name, "direction": "desc"}]
        args.setdefault("limit", 5)
    elif shape == "count":
        if slots.metric:
            # A metric with a counting question is an aggregate ("the total
            # budget"), and this builder cannot express sum/avg — the deep path
            # can. Never serve a ranking in its place.
            return declined("count_with_metric")
        # A count: `total` is the real DB count and the aggregate row makes it
        # citable evidence. Never count rows in Python.
        args["aggregate"] = {"count": {"op": "count"}}
    else:
        # "which/list" wants rows, an aggregate wants sum/avg, and anything
        # unrecognised is a shape this builder has not been shown to get right.
        # All three cost the deep path, per `schemas/router.py`: "an unset flag
        # must cost the deep path, never a confidently wrong count."
        return declined("shape_other")

    return args


# ── the lookup ────────────────────────────────────────────────────────────────


@dataclass
class Lookup:
    """What one deterministic lookup produced: tool results to index (in call
    order), or a ready payload for the 403/404 shortcut. `None` from
    `_lookup` means "deep path"."""

    evidence: list[tuple[str, dict]] = field(default_factory=list)
    payload: AnswerPayload | None = None
    tool: str = ""


def _is_error(response: Any) -> bool:
    return isinstance(response, dict) and bool(response.get("error"))


def _shortcut(
    response: dict,
    decision: RouterDecision,
    shim: _ToolShim,
    *,
    bot_supplied: bool = False,
) -> Lookup | None:
    """A 403/404 answered on the spot; any other error → deep path.

    `bot_supplied` is an id that came from the bot's prefix rather than from
    this path's own search: a 403/404 on it goes to the deep path too, with a
    line of its own. It does NOT write the `shortcut` line, which counts the
    turns this function answered."""
    status = response.get("status")
    if bot_supplied and status in (403, 404):
        logger.info(
            "fast_path: bot-supplied id refused status=%s entity_type=%s intent=%s — deep path"
            " (inv=%s) tenant=%s",
            status,
            decision.entity_type or "-",
            decision.intent,
            *_where(shim),
        )
        return None
    if status in (403, 404):
        logger.info(
            "fast_path: shortcut status=%s intent=%s inv=%s tenant=%s",
            status,
            decision.intent,
            *_where(shim),
        )
    if status == 403:
        return Lookup(payload=no_access_payload(decision.language, decision.entity_surface))
    if status == 404:
        return Lookup(payload=not_found_payload(decision.language, decision.entity_surface))
    logger.info(
        "fast_path: %s from the lookup — deep path (inv=%s) tenant=%s", status, *_where(shim)
    )
    return None


def _campaign_id(decision: RouterDecision) -> str | None:
    """The campaign id the bot's prefix supplied — or None.

    `entity_id` is the bot's "User selected …" / "Best guess: …" id (blend 02),
    and it names a record of whatever type the reader or the bot settled on:
    a piece, an account, a person. Only the prefix's own word for the type
    says which, and `decision_from` copies that word to `entity_type` (None
    when it could not vouch for the id). So an id is a campaign id only when
    the type says campaign; asked of a piece's id, `/org/campaigns/<id>` is a
    404 for a record that exists, which is how the reader came to be told
    "not found" about a piece they had just picked (2026-10-08)."""
    if decision.entity_id and decision.entity_type == CAMPAIGN_TYPE:
        return decision.entity_id
    return None


async def _campaign(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    campaign_id = _campaign_id(decision)
    if decision.entity_id and campaign_id is None:
        # The bot handed over something that is not (known to be) a campaign.
        # Neither fetching it as one NOR searching the surface instead is
        # right: the search could settle on a campaign the reader did not pick.
        # The executor gets the prefix, with its type, and chooses the tool.
        logger.info(
            "fast_path: bot-supplied id is not a campaign entity_type=%s intent=%s — deep path"
            " (inv=%s) tenant=%s",
            decision.entity_type or "-",
            decision.intent,
            *_where(shim),
        )
        return None
    supplied = campaign_id is not None
    if campaign_id is None:
        resolution = await _resolve(decision.entity_surface, ("campaign",), shim)
        if resolution.kind != "one" or resolution.entity_id is None:
            return None
        campaign_id = resolution.entity_id
    response = await get_campaign(campaign_id, shim)
    if _is_error(response):
        return _shortcut(response, decision, shim, bot_supplied=supplied)
    if not isinstance(response, dict) or not response.get("id"):
        return None
    return Lookup(evidence=[("get_campaign", response)], tool="get_campaign")


async def _account(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    resolution = await _resolve(decision.entity_surface, ("account",), shim)
    if resolution.kind != "one" or resolution.entity_id is None:
        return None
    response = await get_account_overview(resolution.entity_id, shim)
    if _is_error(response):
        return _shortcut(response, decision, shim)
    if not isinstance(response, dict) or not response.get("id"):
        return None
    return Lookup(evidence=[("get_account_overview", response)], tool="get_account_overview")


# A question about a person's TIME — bookings, availability, leave — rather
# than who they are. On the /ask engine, where resourcing is a source, the answer
# lives in the planner, which the fast path's one UB lookup never reads: a
# profile would answer "is Marcus free?" with his job title, or worse, as
# "nothing booked". Those turns go to the deep path, which has the planner
# tools. Run 1 of the Runn eval (#114): all seven such questions failed here.
_STAFFING = re.compile(
    r"\b(free|availab\w*|unavailab\w*|booked|overbooked|underbooked|booking\w*|busy|"
    r"capacity|allocat\w*|staffed|scheduled?|utili[sz]\w*|bench|tentative|"
    r"leave|holiday\w*|vacation|time off|day off|days off|out of office|ooo|"
    r"back from|back on|working on|workload|hours|part[- ]time|"
    r"on (?:this|next|that) (?:week|month)|"
    r"(?:people|who|who's|anyone|everyone)\s+(?:is\s+|are\s+)?(?:on|staffed)\b|"
    r"(?:off|out) (?:on|next|this|over|until|till|today|tomorrow)|"
    r"cover for|take on|pick up|start on)",
    re.IGNORECASE,
)


def is_staffing_question(question: str) -> bool:
    """Is this about how a person's time is planned (see `_STAFFING`)?"""
    return bool(_STAFFING.search(question or ""))


#: A message this short is read as a follow-up to the one before it.
FOLLOW_UP_MAX_WORDS = 8


def _declines_staffing(question: str, shim: Any = None) -> bool:
    """On the /ask engine: a staffing question, or a short follow-up to one
    ("What's Jane on next week?" → "Jane Park, in LA.")."""
    if not config.ASK_CONTRACT:
        return False
    if is_staffing_question(question):
        return True
    prior = getattr(shim, "prior_question", "") or ""
    return len(question.split()) <= FOLLOW_UP_MAX_WORDS and is_staffing_question(prior)


async def _staff(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    if _declines_staffing(question, shim):
        logger.info("fast_path: declined staffing question (inv=%s) tenant=%s", *_where(shim))
        return None
    resolution = await _resolve(decision.entity_surface, ("staff",), shim)
    if resolution.kind == "ambiguous":
        return None
    if resolution.kind == "one" and resolution.entity_id:
        response = await get_staff_profile(resolution.entity_id, shim)
        if _is_error(response):
            return _shortcut(response, decision, shim)
        if not isinstance(response, dict) or not response.get("id"):
            return None
        return Lookup(evidence=[("get_staff_profile", response)], tool="get_staff_profile")

    # Nothing in the cross-entity search: `search_staff` also matches titles,
    # emails and departments, which `/org/search` does not — one more
    # deterministic call before giving the turn to the executor.
    if not decision.entity_surface:
        return None
    response = await search_staff(decision.entity_surface, tool_context=shim)
    if _is_error(response):
        return _shortcut(response, decision, shim)
    rows = response.get("staff") if isinstance(response, dict) else None
    if not rows:
        return None
    return Lookup(evidence=[("search_staff", response)], tool="search_staff")


async def _count_or_rank(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    if _declines_staffing(question, shim):
        # "how many people are on Vela?" counts bookings, not UB staff rows.
        logger.info("fast_path: declined staffing question (inv=%s) tenant=%s", *_where(shim))
        return None
    args = await org_query_args(decision, shim, question)
    if args is None:
        return None
    response = await org_query(**args, tool_context=shim)
    if _is_error(response):
        return _shortcut(response, decision, shim)
    if not isinstance(response, dict) or not response.get("results"):
        return None
    return Lookup(evidence=[("org_query", response)], tool="org_query")


_LOOKUPS = {
    "campaign_status": _campaign,
    "campaign_facts": _campaign,
    "account_facts": _account,
    "staff_lookup": _staff,
    "count_or_rank": _count_or_rank,
}


#: Which tool each intent's lookup will reach for, so the bot's progress line
#: can name it BEFORE the call rather than after it — a step that arrives with
#: the answer is not progress. Announcing ahead is accurate: the lookup runs
#: this tool; what it may decline on is the RESULT. A lookup that touches more
#: than one tool is named by the one that does the work.
_FAST_TOOLS = {
    "campaign_status": "get_campaign",
    "campaign_facts": "get_campaign",
    "account_facts": "get_account_overview",
    "staff_lookup": "search_staff",
    "count_or_rank": "org_query",
}


# ── a decline decided in code, before any lookup ──────────────────────────────


async def _resolved_offline(surface: str | None, wanted: tuple[str, ...], shim: Any) -> Resolution:
    """The account name resolves to exactly one id, with no call: the most
    accepting answer GUB could give, so a builder that still declines would
    decline on any answer."""
    return Resolution("one", "precheck", surface)


async def declined_before_lookup(
    decision: RouterDecision, question: str, shim: _ToolShim
) -> str | None:
    """Why the lookup for `decision` would decline whatever GUB answers — a
    reason slug — or None when only a call can tell, or it would answer.

    For the speculative root (SPECULATIVE_FAST_PRECHECK): a fast decision the
    lookup is certain to decline can keep the deep run already started beside
    the router, instead of stopping it for a lookup that hands the turn back.
    So a slug is returned only where it cannot be wrong, each rule one the
    lookup itself applies before any response, or regardless of it:

    - account_facts: `_account` resolves the surface and nothing else (its
      `entity_id` is ignored), and `_resolve` answers "none" for a blank
      surface without a call;
    - staff_lookup: a FALSY surface only — a whitespace-only one still reaches
      `search_staff`, whose answer nobody can predict — or, on the /ask
      engine, a question about the person's time (`is_staffing_question`),
      which `_staff` declines before any call (as `_count_or_rank` does);
    - campaign_*: a bot-supplied id whose type is not campaign (`_campaign`
      hands the turn over before any call) — "entity_type" when the prefix
      names the type, "entity_type_unknown" when no prefix vouched for the id —
      or no id and a blank surface (a campaign id is fetched as is);
    - count_or_rank: `org_query_args` itself, with the account name resolved
      offline (`_resolved_offline`). Its other checks read the decision and
      the question alone, and a name that does not resolve declines as well,
      so a None here is a None whatever `/org/search` returns.

    Keyed on the lookup `_LOOKUPS` holds, not on the intent's name: it speaks
    only for the functions it restates. Pure — no call, no state write; the
    one side effect is the builder's own reason line for incomplete slots or
    an office, which the lookup would log the same way."""
    lookup = _LOOKUPS.get(decision.intent)
    surface = decision.entity_surface
    blank = not surface or not surface.strip()
    if lookup is _account:
        return "no_surface" if blank else None
    if lookup is _staff:
        if _declines_staffing(question, shim):
            return "staffing"
        return "no_surface" if not surface else None
    if lookup is _campaign:
        if decision.entity_id:
            if _campaign_id(decision):
                return None
            return TYPED_ID_REASON if decision.entity_type else UNVOUCHED_ID_REASON
        return "no_surface" if blank else None
    if lookup is _count_or_rank:
        if _declines_staffing(question, shim):
            return "staffing"
        why: list[str] = []
        args = await org_query_args(decision, shim, question, resolve=_resolved_offline, why=why)
        if args is not None:
            return None
        return why[-1] if why else "query_args"
    return None


def log_declined_before_lookup(shim: _ToolShim, reason: str) -> None:
    """The line for a fast decision `declined_before_lookup` settled; it
    follows the turn's `dispatcher: fast path declined` line."""
    logger.info(
        "fast_path: declined before lookup reason=%s (inv=%s) tenant=%s", reason, *_where(shim)
    )


# ── the draft the format gate renders ─────────────────────────────────────────


def _hidden_id(key: str, value: Any) -> bool:
    """READER_TEXT_SCRUB: a record's own id stays out of the draft. The draft is
    what the formatter writes the answer from, and it wrote "(id: <uuid>)" into
    a reader's reply because the draft listed one. The id is still in the
    evidence rows (`ALLOWED_EVIDENCE`), where it is cited and echoed."""
    return config.READER_TEXT_SCRUB and is_record_id_field(key, value)


def _scalar_lines(row: dict) -> list[str]:
    lines = []
    for key, value in row.items():
        if key.startswith("_") or not isinstance(value, (str, int, float, bool)):
            continue
        if _hidden_id(key, value):
            continue
        text = str(value)
        if len(text) > MAX_DRAFT_FIELD_CHARS:
            text = text[:MAX_DRAFT_FIELD_CHARS] + "…"
        lines.append(f"{key}: {text}")
    return lines


def compose_draft(question: str, decision: RouterDecision, lookup: Lookup) -> str:
    """The gate's input for a fast-path turn — the question and the fields the
    one call returned, copied verbatim. Values are never reformatted here: the
    gate grounds every number against the indexed value, and a prettified
    number would be rejected as ungrounded (correctly)."""
    lines = [
        f"DETERMINISTIC LOOKUP (fast path) — intent={decision.intent}, tool={lookup.tool}. "
        "Answer the question from these fields; nothing else was retrieved.",
        f"QUESTION: {question.strip()}",
        "RESULT:",
    ]
    if config.READER_TEXT_SCRUB:
        # The formatter prompt's own rule, said again where the fields are.
        lines[0] += (
            " Never write a record id or UUID, and write a date as a human date"
            " (June 4, 2026), never as an ISO timestamp."
        )
    for _tool, response in lookup.evidence:
        lines.extend(_scalar_lines(response))
        for key, value in response.items():
            if key.startswith("_") or not isinstance(value, list):
                continue
            rows = [item for item in value if isinstance(item, dict)][:MAX_DRAFT_ROWS]
            if not rows:
                continue
            lines.append(f"{key}:")
            for item in rows:
                scalars = {
                    k: v
                    for k, v in item.items()
                    if isinstance(v, (str, int, float, bool))
                    and not k.startswith("_")
                    and not _hidden_id(k, v)
                }
                lines.append(f"- {json.dumps(scalars, ensure_ascii=False, default=str)}")
    return "\n".join(lines)


# ── the agent ─────────────────────────────────────────────────────────────────


class _ToolStub:
    """`record_evidence` reads `tool.name` to prefix the evidence ids
    (`evidence_index.py:record_evidence`) — using the real tool names keeps a
    fast-path `evidence_id` indistinguishable from a deep-path one, which is
    what blend 05's conflict filter and the eval reader rely on."""

    def __init__(self, name: str) -> None:
        self.name = name


def lookup_tool(lookup: Lookup | None, fallback: str) -> str:
    """The tool a finished lookup says it used, falling back to what was
    announced. `Lookup.tool` is authoritative where it is set — a lookup that
    shortcut to a 403/404 payload may not have reached the tool it planned."""
    return lookup.tool if lookup is not None and lookup.tool else fallback


class FastPath(BaseAgent):
    """One lookup, then the format gate. Emits nothing at all when it decides
    the question is not its business — the dispatcher reads `outcome()` and
    runs the deep path."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        _set_outcome(ctx.invocation_id, "deep")  # pessimistic until a payload ships
        decision = decision_from(ctx)
        shim = _ToolShim(ctx)
        # `Slots` cannot say whether the sentence wanted a count, a list or a
        # sum, so the count_or_rank builder reads the user's own words.
        question = user_text(ctx)

        # The fast path owns this turn's index: the executor's per-pass reset
        # (`agent.py:_before_agent`) has not run and must not be relied on.
        reset_evidence_index(shim)

        lookup_fn = _LOOKUPS.get(decision.intent)
        if lookup_fn is None:
            return

        # Say what is about to happen. Without this the bot has no tool
        # activity to show and its bubble holds one generic phrase for the
        # whole turn, which reads as a hang — the deep path never had the
        # problem because its calls go through a model and emit function_call
        # parts of their own. These events are `partial`, so they reach the
        # bot and never enter the session history (see `tool_activity_event`).
        announced = _FAST_TOOLS.get(decision.intent)
        if announced:
            yield tool_activity_event(ctx, announced)
        lookup: Lookup | None = None
        raised = False
        try:
            lookup = await lookup_fn(decision, shim, question)
        except Exception:
            # Never fail the turn on the fast path — the deep path is always a
            # correct (if slower) answer to the same question.
            logger.exception(
                "fast_path: lookup raised (inv=%s) — deep path tenant=%s",
                ctx.invocation_id,
                label_of(ctx),
            )
            raised = True
        # Closed even when the lookup raised or declined. The pair is not
        # decoration: the bot counts calls per tool to number repeats, and a
        # call left open holds the bubble on "Querying org records…" while the
        # deep path is already working underneath it.
        if announced:
            yield tool_activity_event(ctx, lookup_tool(lookup, announced), done=True)
        if raised or lookup is None:
            return

        if lookup.payload is not None:
            _set_outcome(ctx.invocation_id, "answered")
            yield payload_event(ctx, lookup.payload)
            return

        for tool_name, response in lookup.evidence:
            record_evidence(_ToolStub(tool_name), {}, shim, response)
        index = evidence_index(ctx.invocation_id)
        if not index:
            # Nothing citable came back — a payload would have to be
            # uncited, which is exactly what the contract forbids.
            logger.info(
                "fast_path: no citable evidence (inv=%s) — deep path tenant=%s",
                ctx.invocation_id,
                label_of(ctx),
            )
            return

        set_answer_draft(ctx.invocation_id, compose_draft(question, decision, lookup))
        _set_outcome(ctx.invocation_id, "answered")
        # The gate is the deep path's own object, invoked here rather than
        # adopted: ADK allows an agent ONE parent (`base_agent.py:135-145`),
        # and `run_async` needs nothing but a context. Same gate, same
        # validators, same retry-then-template guarantee — so the fast path
        # cannot ship a payload the deep path would have rejected.
        async for event in format_gate.run_async(ctx):
            yield event


fast_path = FastPath(name="fast_path")
