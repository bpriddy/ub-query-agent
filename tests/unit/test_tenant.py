"""
tenant — which branded surface a turn arrived through (gub-gchat-bot#83/#90).

One engine, more than one Chat app. The load-bearing case is the NEGATIVE one:
a turn with no `tenant` in state must get the base instruction back byte for
byte, because that is every production turn the original bot has ever sent and
will keep sending. Case 1 pins that against the real ADK instruction path, not
a fake.

The positive cases pin the two things the tenancy ruling actually bought: the
scope block reaching the prompt with the label interpolated (never hardcoded —
this engine is multi-company), and every per-turn log line carrying a `tenant=`
field so log-derived proportions stop mixing two bots' traffic.

The loud-failure case matters more than it looks: a bot that believes it is
scoped and is not is exactly the state nobody notices.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gub_agent.tenant import (
    DEFAULT_LABEL,
    label_of,
    read_tenant,
    tenant_instruction,
    tenant_note,
    tenant_note_formatter,
)


def _ctx(state: dict | None) -> SimpleNamespace:
    """An instruction-provider-shaped context."""
    return SimpleNamespace(state=state)


#: A tenant turn always carries its app scope (a string); synthetic label.
_APP_SCOPE = '{"account":[{"id":"11111111-1111-1111-1111-111111111111","name":"Acme"}]}'


def _inv_ctx(state: dict | None) -> SimpleNamespace:
    """An InvocationContext-shaped context (state hangs off `.session`)."""
    return SimpleNamespace(session=SimpleNamespace(state=state))


# ── 1. the negative case: no tenant key, nothing happens ────────────────────


def test_no_tenant_returns_the_base_text_unchanged() -> None:
    base = "BASE INSTRUCTION\n\n## Current date\nToday is 2026-09-17 (UTC)."
    provider = tenant_instruction(lambda _ctx: base)

    assert provider(_ctx({"gub_jwt": "jwt"})) == base
    assert provider(_ctx({})) == base
    assert provider(_ctx(None)) == base


def test_no_tenant_logs_the_default_label() -> None:
    assert label_of(_ctx({"gub_jwt": "jwt"})) == DEFAULT_LABEL
    assert label_of(_inv_ctx({})) == DEFAULT_LABEL
    assert label_of(None) == DEFAULT_LABEL


# ── 2. the positive case: the scope block, with the label interpolated ──────


def test_tenant_appends_the_scope_block() -> None:
    provider = tenant_instruction(lambda _ctx: "BASE")
    out = provider(_ctx({"gub_jwt": "jwt", "tenant": "chevy", "app_scope": _APP_SCOPE}))

    assert out.startswith("BASE\n\n")
    assert "## Tenant surface" in out
    assert "chevy" in out


def test_the_block_names_no_client() -> None:
    """The engine is multi-company: the label is interpolated, never baked in.
    A client name in this prompt is a client name in every tenant's prompt."""
    note = tenant_note("someclient")

    assert note.count("someclient") >= 3
    assert "chevy" not in note.lower()
    assert "anomaly" not in note.lower()


def test_the_block_no_longer_denies_the_boundary() -> None:
    """tenant-10 reversed the ruling: the 📊 half IS restricted now, enforced
    in the backend. The old text said the opposite in two places, and a model
    told 'you are not restricted' will argue with its own tool results."""
    note = tenant_note("someclient").lower()

    assert "not restricted" not in note
    assert "never say or imply that other accounts are hidden" not in note


def test_the_block_binds_the_claim_to_the_evidence() -> None:
    """Withheld records are described only when a tool result says so — which
    keeps the instruction correct in BOTH enforcement modes, since 'log' mode
    withholds nothing and emits no notice."""
    note = tenant_note("someclient")

    assert "account_scope_notice" in note
    assert "does not exist" in note.lower() or "not exist" in note.lower()
    # And it forbids the other wrong answer: blaming the user's own access.
    assert "lacks access" in note.lower()


def test_the_workspace_half_is_called_out_as_unscoped() -> None:
    """The ✉️ half is not scoped and cannot be with today's machinery. A model
    that thinks everything is scoped will describe mailbox results as limited."""
    note = tenant_note("someclient").lower()

    assert "not scoped" in note


def test_the_formatter_block_is_its_own_and_narrower() -> None:
    """The formatter is the ONLY model on fast-path, smalltalk, abstain and
    clarify turns, so it needs the rule directly — but it sees no tool
    results, so its job is to not lose a notice rather than to raise one."""
    note = tenant_note_formatter("someclient")

    assert "## Tenant surface" in note
    assert "someclient" in note
    # It must NOT keep a scope sentence: the surface prepends exactly one,
    # and a second reads as a stutter (four renderings on one live turn).
    assert "do not add a sentence saying records were withheld" in note.lower()
    # The format-gate trap: a capitalised label in no tool result is an
    # ungrounded entity and burns all three formatter attempts.
    assert "do not add the surface's name" in note.lower()


def test_the_formatter_block_is_opt_in_like_the_executor_one() -> None:
    provider = tenant_instruction(lambda _ctx: "BASE", note=tenant_note_formatter)

    assert provider(_ctx({"gub_jwt": "jwt"})) == "BASE"
    assert "## Tenant surface" in provider(_ctx({"tenant": "someclient", "app_scope": _APP_SCOPE}))


def test_state_read_works_through_both_context_shapes() -> None:
    assert label_of(_ctx({"tenant": "chevy"})) == "chevy"
    assert label_of(_inv_ctx({"tenant": "chevy"})) == "chevy"


# ── 3. a malformed label is loud, never a silent fall back to unscoped ──────


@pytest.mark.parametrize("bad", ["Chevy", "chevy bot", "", "-chevy", "x" * 33, 7, ["chevy"]])
def test_malformed_label_raises(bad: object) -> None:
    with pytest.raises(ValueError):
        read_tenant({"tenant": bad})


def test_log_labelling_never_raises_on_a_bad_label() -> None:
    """`label_of` is called from log statements. A bad label must not turn a
    log line into a crashed turn — the instruction path already raised."""
    assert label_of(_ctx({"tenant": "NOT A SLUG"})) == DEFAULT_LABEL


# ── 4. the log lines the CLAUDE.md queries actually match on ────────────────


def test_dispatcher_line_keeps_its_existing_fields_and_gains_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The `dispatcher: intent` line is the denominator of every per-turn
    proportion measured from the logs. `tenant=` is appended so the existing
    substring matches keep working."""
    from gub_agent.agents import dispatcher as dispatcher_module

    logger = logging.getLogger(dispatcher_module.__name__)
    with caplog.at_level(logging.INFO, logger=dispatcher_module.__name__):
        logger.info(
            "dispatcher: intent=%s confidence=%.2f branch=%s (inv=%s) tenant=%s",
            "count_or_rank",
            0.91,
            "fast",
            "inv-1",
            label_of(_inv_ctx({"tenant": "chevy"})),
        )

    line = caplog.messages[-1]
    assert line.startswith("dispatcher: intent=count_or_rank confidence=0.91 branch=fast")
    assert "(inv=inv-1)" in line
    assert line.endswith("tenant=chevy")


# ── 6. BE-7: tenant-aware persona ───────────────────────────────────────────

import hashlib  # noqa: E402

from gub_agent import agent as _agent  # noqa: E402
from gub_agent.agents import critic as _critic  # noqa: E402
from gub_agent.agents import router as _router  # noqa: E402
from gub_agent.prompts import (  # noqa: E402
    CRITIC_INSTRUCTION,
    EXECUTOR_INSTRUCTION,
    ROUTER_INSTRUCTION,
    TENANT_EXECUTOR_INSTRUCTION,
)

# sha256 of the base prompts as they were BEFORE the tenant persona landed.
# The executor pin is the base after the format-rule fix that landed on the same
# branch; the tenant persona itself does not change the no-tenant text.
# Re-pinned 2026-10-29: campaign status list gained in_progress and paused.
# Re-pinned 2026-11-18: text operators starts_with / ends_with / contains (#100).
_BASE_SHA = {
    "executor": "06fed3ee3a314251d77c869b1afcc954c3382e3e8b95691038ee6b2c2b9f546f",
    "router": "4f1cd2536b86bcd705e72173479e5c855fe9a5f060be5cf660f3de8095a0b33e",
    "critic": "83b3ff0bc22ced495e76c7376e5833bf961e500a78a90359c98376e7f993e0f4",
}
_SCOPE = "## Scope — decide first"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _render(agent, state: dict) -> str:
    ctx = SimpleNamespace(state=state, session=SimpleNamespace(state=state, events=[]))
    return agent.instruction(ctx)


def test_base_prompts_are_byte_identical_to_pre_tenant_persona() -> None:
    assert _sha(EXECUTOR_INSTRUCTION) == _BASE_SHA["executor"]
    assert _sha(ROUTER_INSTRUCTION) == _BASE_SHA["router"]
    assert _sha(CRITIC_INSTRUCTION) == _BASE_SHA["critic"]


@pytest.mark.parametrize(
    "agent", [_agent.executor_agent, _router.router_agent, _critic.critic_agent]
)
def test_no_tenant_renders_no_tenant_text(agent) -> None:
    out = _render(agent, {"gub_jwt": "jwt"})
    assert "## Tenant surface" not in out
    assert "one client account's team" not in out


def test_no_tenant_executor_keeps_the_agency_persona() -> None:
    out = _render(_agent.executor_agent, {"gub_jwt": "jwt"})
    assert out.startswith(EXECUTOR_INSTRUCTION)


@pytest.mark.parametrize("agent", [_router.router_agent, _critic.critic_agent])
def test_router_and_critic_carry_the_scope_block_on_a_tenant_turn(agent) -> None:
    out = _render(agent, {"gub_jwt": "jwt", "tenant": "chevy", "app_scope": _APP_SCOPE})
    assert "## Tenant surface" in out
    assert "chevy" in out


def test_tenant_executor_swaps_the_agency_persona() -> None:
    out = _render(
        _agent.executor_agent, {"gub_jwt": "jwt", "tenant": "chevy", "app_scope": _APP_SCOPE}
    )
    assert "one client account's team" in out
    assert "## Tenant surface" in out
    for gone in (
        "You are GUB AI",
        "agency operations",
        "find_staff_for_resourcing",
        "get_staff_profile",
        "resourcing requests",
        "GUB indexes",
    ):
        assert gone not in out, gone


def test_tenant_executor_keeps_tool_use_and_safety_rules() -> None:
    out = _render(
        _agent.executor_agent, {"gub_jwt": "jwt", "tenant": "chevy", "app_scope": _APP_SCOPE}
    )
    base_tail = EXECUTOR_INSTRUCTION.partition(_SCOPE)[2]
    for rule in (
        "GROUND EVERY ENTITY IN A TOOL RESULT",
        "NO_COMPANY_RECORDS",
        "Never expose internal identifiers",
        "FAN OUT",
        "`org_query` filter reference",
    ):
        assert rule in out, rule
        assert rule in base_tail, rule


def test_tenant_persona_uses_neutral_vocabulary() -> None:
    persona = TENANT_EXECUTOR_INSTRUCTION.partition(_SCOPE)[0]
    for banned in ("GUB", "Universal Backend", "Gemini", "federated", "spoke", "agency"):
        assert banned not in persona, banned


def test_tenant_prompt_has_no_internal_names_or_staff_directory_text() -> None:
    for banned in (
        "GUB",
        "Universal Backend",
        "Gemini",
        "federated",
        "spoke",
        "on_leave",
        "Staff who led",
        "staff-by-id",
    ):
        assert banned not in TENANT_EXECUTOR_INSTRUCTION, banned
    assert "ideas, or staff" not in TENANT_EXECUTOR_INSTRUCTION


def test_tenant_executor_prompt_names_no_client_until_the_block() -> None:
    assert "chevy" not in TENANT_EXECUTOR_INSTRUCTION.partition(_SCOPE)[0].lower()


def test_a_malformed_tenant_still_fails_loud_on_the_executor() -> None:
    with pytest.raises(ValueError):
        _render(_agent.executor_agent, {"gub_jwt": "jwt", "tenant": "Chevy!"})
