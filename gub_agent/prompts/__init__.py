"""
prompts/ — the agent's instruction text, extracted from code for hand-editing.

Each prompt lives in its own module as a plain triple-quoted string so it can
be edited as prose without touching the wiring in `agent.py` / `agents/`. These
are `.py` (not `.md`) on purpose: `adk deploy agent_engine` packages Python
modules reliably, whereas loose data files can be dropped from the bundle.

Two things to keep in mind when editing any prompt here:

1. Curly braces are safe HERE, but only because of how these strings are
   wired. ADK runs a state-substitution pass (`re.sub('{+[^{}]*}+', ...)`) over
   instruction strings, and a `{...}` that isn't a real session-state key
   crashes the agent with a KeyError — visible only in the Reasoning Engine
   logs, symptom is an empty reply. That pass is SKIPPED for callable
   instructions (`canonical_instruction` reports `bypass_state_injection=True`
   for an InstructionProvider), and both agents wrap these strings in one
   (`instruction_utils.with_current_date`, `sandbox.sandbox_instruction`). The
   trap returns the moment an `instruction=` is handed a plain string — so if
   you unwrap one, grep the prompt for `{` first.
2. Don't hardcode a date. The current date is injected per-request by
   `with_current_date()` (see `instruction_utils.py`) and appended to whatever
   string you write here.

The modules here are what PRODUCTION runs. Experimental rewrites live in
`variants/` and are selected per call by name (`state["sandbox"]`); a variant
that wins is promoted into this file by hand, in an ordinary pull request.
"""

from .critic import CRITIC_INSTRUCTION
from .executor import EXECUTOR_INSTRUCTION, RUNN_GUIDANCE, TENANT_EXECUTOR_INSTRUCTION
from .formatter import FORMATTER_INSTRUCTION
from .router import ROUTER_INSTRUCTION

__all__ = [
    "CRITIC_INSTRUCTION",
    "EXECUTOR_INSTRUCTION",
    "FORMATTER_INSTRUCTION",
    "ROUTER_INSTRUCTION",
    "RUNN_GUIDANCE",
    "TENANT_EXECUTOR_INSTRUCTION",
]
