"""
The query agent service's `/ask` (ub-query-agent#92).

Pinned here:

  * who is asking comes from the UB token only: the person is its subject,
    the app its per-app audience; no token or a bad one is refused;
  * a conversation is stored with its actor and context, and resumed only by
    the same person in the same app within the idle limit; someone else's id
    and an unknown id get the same 404; an expired one starts a new one;
  * the stream is conversation, progress, then exactly one answer, carrying
    our conversation id; only `ask`-authored events are relayed;
  * a failed or answerless turn still ends with an error answer that is a
    valid `AskResponse`;
  * the caller's UB token and the minted Google token reach the engine in
    `temp:` state and appear in no stored row.
"""

from __future__ import annotations

from gub_agent.schemas.ask import AskResponse
from query_service.app import error_answer
from query_service.engine import UB_TOKEN_KEY, WORKSPACE_TOKEN_KEY
from query_service.grants import put_grant

from .conftest import APP, CLIENT_ID, FakeCipher, ask, parse_sse, ub_token


async def test_a_turn_streams_conversation_progress_then_one_answer(service):
    token = ub_token()
    resp = await ask(service, token, actor="ava", context={"type": 1, "id": "holocron"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert [name for name, _ in events] == ["conversation", "progress", "answer"]
    conversation = events[0][1]
    assert conversation["new"] is True
    answer = AskResponse.model_validate(events[-1][1])
    assert answer.conversation_id == conversation["conversation_id"]
    assert "secret-row" not in resp.text  # the engine's own events stay here

    row = await service.pool.fetchrow("SELECT * FROM conversations")
    assert (
        row["person_id"],
        row["app_id"],
        row["actor"],
        row["context_type"],
        row["context_id"],
    ) == (
        "person-1",
        APP,
        "ava",
        1,
        "holocron",
    )
    [turn] = service.engine.turns
    assert turn["user_id"] == "person-1"
    assert turn["session_id"] == row["engine_session_id"]
    assert turn["state"] == {UB_TOKEN_KEY: token}


async def test_the_owner_resumes_their_conversation(service):
    first = parse_sse((await ask(service, ub_token())).text)[0][1]["conversation_id"]
    resp = await ask(service, ub_token(), conversation_id=first)

    events = parse_sse(resp.text)
    assert events[0][1] == {"conversation_id": first, "new": False}
    assert len(service.engine.sessions) == 1
    assert service.engine.turns[1]["session_id"] == service.engine.turns[0]["session_id"]


async def test_another_person_or_app_or_an_unknown_id_gets_the_same_404(service):
    first = parse_sse((await ask(service, ub_token())).text)[0][1]["conversation_id"]

    for token, cid in [
        (ub_token(person="person-2"), first),
        (ub_token(app="another-app"), first),
        (ub_token(app=None), first),
        (ub_token(), "00000000-0000-4000-8000-000000000000"),
        (ub_token(), "not-a-uuid"),
    ]:
        resp = await ask(service, token, conversation_id=cid)
        assert resp.status_code == 404
        assert resp.json() == {"code": "CONVERSATION_NOT_FOUND", "message": "No such conversation."}
    assert len(service.engine.turns) == 1


async def test_actor_and_context_are_not_part_of_ownership(service):
    first = parse_sse((await ask(service, ub_token(), actor="ava")).text)[0][1]["conversation_id"]
    resp = await ask(service, ub_token(), conversation_id=first, context={"type": 2})
    assert parse_sse(resp.text)[0][1] == {"conversation_id": first, "new": False}


async def test_an_expired_conversation_starts_a_new_one(service):
    first = parse_sse((await ask(service, ub_token())).text)[0][1]["conversation_id"]
    await service.pool.execute(
        "UPDATE conversations SET last_active_at = now() - interval '16 minutes'"
    )

    events = parse_sse((await ask(service, ub_token(), conversation_id=first)).text)
    second = events[0][1]
    assert second["new"] is True and second["conversation_id"] != first
    assert events[-1][1]["conversation_id"] == second["conversation_id"]
    assert len(service.engine.sessions) == 2


async def test_no_token_or_a_bad_one_is_refused_before_anything_runs(service):
    assert (await ask(service, None)).json()["code"] == "MISSING_TOKEN"
    for bad in [
        ub_token(iss="https://elsewhere.test"),
        ub_token(aud="holocron"),  # signed for an app, not for UB
        ub_token(exp=1),
        ub_token() + "x",
    ]:
        resp = await ask(service, bad)
        assert resp.status_code == 401 and resp.json()["code"] == "INVALID_TOKEN"
    assert service.engine.sessions == []


async def test_a_bad_body_is_a_400(service):
    resp = await service.client.post(
        "/ask", headers={"Authorization": f"Bearer {ub_token()}"}, json={"question": ""}
    )
    assert resp.status_code == 400 and resp.json()["code"] == "BAD_REQUEST"
    resp = await ask(service, ub_token(), person="someone-else")
    assert resp.status_code == 200  # an unknown field is ignored, never read


async def test_a_turn_without_an_answer_ends_with_an_error_answer(service):
    service.engine.script = service.engine.script[:2]
    events = parse_sse((await ask(service, ub_token())).text)
    answer = AskResponse.model_validate(events[-1][1])
    assert (answer.kind, answer.error.code) == ("error", "NO_ANSWER")
    assert answer.conversation_id == events[0][1]["conversation_id"]


async def test_a_turn_that_fails_midway_still_ends_with_an_answer(service):
    service.engine.script = service.engine.script[:1]
    service.engine.fail_with = RuntimeError("socket closed")
    events = parse_sse((await ask(service, ub_token())).text)
    assert [n for n, _ in events] == ["conversation", "progress", "answer"]
    answer = AskResponse.model_validate(events[-1][1])
    assert answer.error.code == "INTERNAL"
    assert "socket closed" not in events[-1][1]["error"]["message"]


def test_the_error_answer_is_a_valid_ask_response():
    AskResponse.model_validate(error_answer("X", "y", "c1"))


async def test_a_grant_mints_a_google_token_for_the_turn(service):
    await put_grant(
        service.pool, FakeCipher(), "person-1", "1//refresh-token", ["scope"], CLIENT_ID
    )
    await ask(service, ub_token())
    assert service.engine.turns[0]["state"][WORKSPACE_TOKEN_KEY] == "ya29.minted-google-token"
    [call] = service.google.calls
    assert call["refresh_token"] == "1//refresh-token" and call["client_id"] == CLIENT_ID

    # Someone without a grant gets no Google token.
    await ask(service, ub_token(person="person-2"))
    assert WORKSPACE_TOKEN_KEY not in service.engine.turns[1]["state"]


async def test_a_revoked_or_foreign_grant_mints_nothing(service):
    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", ["scope"], CLIENT_ID)
    service.google.status = 400
    await ask(service, ub_token())
    assert WORKSPACE_TOKEN_KEY not in service.engine.turns[0]["state"]

    await put_grant(service.pool, FakeCipher(), "person-1", "1//r", ["scope"], "other-client")
    service.google.status = 200
    await ask(service, ub_token())
    assert WORKSPACE_TOKEN_KEY not in service.engine.turns[1]["state"]
    assert len(service.google.calls) == 1


async def test_no_token_appears_in_any_stored_row(service):
    token = ub_token()
    await put_grant(service.pool, FakeCipher(), "person-1", "1//refresh-token", ["s"], CLIENT_ID)
    first = parse_sse((await ask(service, token)).text)[0][1]["conversation_id"]
    await ask(service, token, conversation_id=first)

    tables = [
        r["tablename"]
        for r in await service.pool.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
    ]
    dump = ""
    for table in tables:
        rows = await service.pool.fetch(f'SELECT t::text AS row FROM "{table}" t')
        dump += "\n".join(r["row"] for r in rows)
    assert "conversations" in tables and "workspace_grants" in tables
    for secret in [token, token.split(".")[2], "ya29.minted-google-token", "1//refresh-token"]:
        assert secret not in dump
