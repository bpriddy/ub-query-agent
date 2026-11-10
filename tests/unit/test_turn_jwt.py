"""
The caller's GUB JWT for one turn (`temp:gub_jwt`, sent by the query agent
service, #92).

Pinned here, on a real Runner: the tools use it for the turn, it wins over a
cached `gub_jwt`, and it is in no stored event or state afterwards, including
the `gub_jwt` cache the other token paths write to.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.adk.sessions.state import State
from google.genai import types as genai_types

from gub_agent.tools._client import TURN_JWT_STATE_KEY, _resolve_gub_jwt

TOKEN = "eyJ.turn-only-ub-token.sig"


class _UsesTheToken(BaseAgent):
    seen: list[str] = []

    async def _run_async_impl(self, ctx):
        # A tool's state: reads see the session, writes land in the event's
        # state_delta and are saved with it, as a real ToolContext's are.
        actions = EventActions()
        state = State(value=ctx.session.state, delta=actions.state_delta)
        jwt = await _resolve_gub_jwt(SimpleNamespace(state=state))
        type(self).seen.append(jwt)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=actions,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text="done")]),
        )


async def test_the_turn_token_is_used_and_never_saved():
    _UsesTheToken.seen = []
    runner = InMemoryRunner(agent=_UsesTheToken(name="gub_root"), app_name="gub")
    session = await runner.session_service.create_session(
        app_name="gub", user_id="u", state={"gub_jwt": "an-older-cached-token"}
    )
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="hi")])
    async for _ in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=message,
        state_delta={TURN_JWT_STATE_KEY: TOKEN},
    ):
        pass

    assert _UsesTheToken.seen == [TOKEN]
    stored = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    dump = json.dumps(stored.model_dump(mode="json"))
    assert TOKEN not in dump
    assert stored.state["gub_jwt"] == "an-older-cached-token"
