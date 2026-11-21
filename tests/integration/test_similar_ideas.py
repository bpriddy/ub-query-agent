"""
similar_ideas / similar_to_idea (UB D40) against MockGub: what is SENT (only
the options the model set, in UB's names), the fallback the model reads where
similarity search is off or not configured, and every other error passed
through as `_client` shaped it.
"""

from __future__ import annotations

import copy

from gub_agent.agents.evidence_index import _index_response
from gub_agent.tools._client import _friendly_error
from gub_agent.tools.similarity import (
    NOT_EMBEDDED_MESSAGE,
    UNAVAILABLE_MESSAGE,
    UNAVAILABLE_OR_UNKNOWN_IDEA_MESSAGE,
    similar_ideas,
    similar_to_idea,
)

POST = ("POST", "/org/ideas/similar")
GET = ("GET", "/org/ideas/i1/similar")

GRAPE = {"text": "a huge grape rolls down the street", "weight": 1.0}
INSTAGRAM = {"text": "an Instagram campaign", "weight": 0.2}

ANSWER = {
    "scoring": "blend@1",
    "unembedded": 2,
    "results": [
        {
            "id": "i2",
            "name": "Night Shift",
            "facets": ["A grape mascot tumbles through town", "Paid social on Instagram"],
            "pitchedAt": "2026-03-01T00:00:00Z",
            "accountName": "Welch's",
            "campaignName": None,
            "awarded": False,
            "score": 0.81,
            "evidence": [
                {"query": 1, "facet": "A grape mascot tumbles through town", "similarity": 0.81},
                {"query": 0, "facet": "Paid social on Instagram", "similarity": 0.93},
            ],
        },
        {
            "id": "i3",
            "name": "Grape Expectations",
            "facets": [],
            "pitchedAt": None,
            "accountName": None,
            "campaignName": None,
            "awarded": True,
            "score": 0.4,
            "evidence": [{"query": 0, "name": "Grape Expectations", "similarity": 0.4}],
        },
    ],
}


def _answer() -> dict:
    return copy.deepcopy(ANSWER)


# ── similar_ideas: the request ────────────────────────────────────────────────


async def test_sends_only_the_facets_when_nothing_else_is_set(gub):
    gub.routes[POST] = (200, _answer())

    await similar_ideas([INSTAGRAM, GRAPE])

    assert gub.requests == [POST]
    assert gub.bodies == [{"facets": [INSTAGRAM, GRAPE]}]


async def test_sends_every_set_option_under_ubs_names(gub):
    gub.routes[POST] = (200, _answer())

    await similar_ideas(
        [GRAPE],
        scoring="blend",
        alpha=0.7,
        search_names=False,
        limit=5,
        account_id="a1",
        campaign_id="c1",
        since="2026-01-01",
        until="2026-06-30",
    )

    assert gub.bodies == [
        {
            "facets": [GRAPE],
            "scoring": "blend",
            "alpha": 0.7,
            "searchNames": False,
            "limit": 5,
            "accountId": "a1",
            "campaignId": "c1",
            "since": "2026-01-01",
            "until": "2026-06-30",
        }
    ]


async def test_facets_carry_only_text_and_a_given_weight(gub):
    """UB refuses unknown facet keys, and a weight the model left out is UB's
    default (1), not something to invent here."""
    gub.routes[POST] = (200, _answer())

    await similar_ideas(
        [
            {"text": "  a huge grape  ", "weight": 0.9, "why": "specific"},
            {"text": "a social campaign"},
            {"text": "   "},
            {"weight": 1.0},
            "a truck that works at night",
        ]
    )

    assert gub.bodies[0]["facets"] == [
        {"text": "a huge grape", "weight": 0.9},
        {"text": "a social campaign"},
        {"text": "a truck that works at night"},
    ]


async def test_no_usable_facet_is_refused_without_a_call(gub):
    result = await similar_ideas([{"text": ""}])

    assert result["error"] is True
    assert result["status"] == 400
    assert gub.requests == []


# ── similar_ideas: the answer ─────────────────────────────────────────────────


async def test_each_result_names_the_match_that_counted_most(gub):
    gub.routes[POST] = (200, _answer())

    result = await similar_ideas([INSTAGRAM, GRAPE])

    assert result["scoring"] == "blend@1"
    assert result["unembedded"] == 2
    assert [r["matchedOn"] for r in result["results"]] == [
        "A grape mascot tumbles through town",
        "Grape Expectations",  # a name match says so
    ]
    assert result["results"][0]["evidence"] == ANSWER["results"][0]["evidence"]


async def test_results_feed_the_evidence_index_as_rows(gub):
    """The answer is a dict with a `results` list, so each similar idea is a
    citable row of the index (an idea, per `ask.TOOL_ENTITY`), with its score
    and what it matched on as field rows."""
    gub.routes[POST] = (200, _answer())
    result = await similar_ideas([INSTAGRAM, GRAPE])

    index: dict = {}
    _index_response("similar_ideas", result, index)

    assert "similar_ideas:i2" in index
    assert index["similar_ideas:i2"]["entity_id"] == "i2"
    assert index["similar_ideas:i2:score"]["value"] == "0.81"
    assert index["similar_ideas:i2:matchedOn"]["value"] == "A grape mascot tumbles through town"
    assert index["similar_ideas:unembedded"]["value"] == "2"


# ── similar_ideas: fallback and errors ────────────────────────────────────────


async def test_feature_off_404_tells_the_model_to_fall_back(gub):
    gub.routes[POST] = (404, {"code": "NOT_FOUND", "message": "Route not found"})

    result = await similar_ideas([GRAPE])

    assert result["error"] is True
    assert result["status"] == 404
    assert result["similarity_unavailable"] is True
    assert result["message"] == UNAVAILABLE_MESSAGE
    assert "list_ideas" in result["message"]


async def test_embeddings_not_configured_503_tells_the_model_to_fall_back(gub):
    gub.routes[POST] = (503, {"code": "EMBEDDINGS_NOT_CONFIGURED", "message": "x"})

    result = await similar_ideas([GRAPE])

    assert result["status"] == 503
    assert result["code"] == "EMBEDDINGS_NOT_CONFIGURED"
    assert result["similarity_unavailable"] is True
    assert result["message"] == UNAVAILABLE_MESSAGE


async def test_the_fallback_is_not_evidence(gub):
    gub.routes[POST] = (404, {"code": "NOT_FOUND"})
    result = await similar_ideas([GRAPE])

    index: dict = {}
    _index_response("similar_ideas", result, index)

    assert index == {}


async def test_other_errors_pass_through_as_the_client_shaped_them(gub):
    path = "/org/ideas/similar"
    for status, body in [
        (403, {}),
        (429, {"code": "RATE_LIMITED"}),
        (502, {"code": "EMBED_FAILED"}),
        (503, {"code": "SOMETHING_ELSE"}),
        (400, {"code": "VALIDATION_ERROR"}),
    ]:
        gub.routes[POST] = (status, body)

        result = await similar_ideas([GRAPE])

        expected = {"error": True, "status": status, "message": _friendly_error(status, path)}
        if "code" in body:
            expected["code"] = body["code"]
        assert result == expected, status


# ── similar_to_idea ───────────────────────────────────────────────────────────


async def test_similar_to_idea_sends_no_unset_option(gub):
    gub.routes[GET] = (200, _answer())

    result = await similar_to_idea("i1")

    assert gub.targets == [("GET", "/org/ideas/i1/similar")]
    assert result["results"][0]["matchedOn"] == "A grape mascot tumbles through town"


async def test_similar_to_idea_sends_set_options_as_query_parameters(gub):
    gub.routes[GET] = (200, _answer())

    await similar_to_idea(
        "i1",
        scoring="strongest",
        search_names=True,
        limit=3,
        account_id="a1",
        since="2026-01-01",
    )
    await similar_to_idea("i1", scoring="blend", alpha=0.25, search_names=False)

    first, second = (t for _, t in gub.targets)
    assert first.startswith("/org/ideas/i1/similar?")
    for part in ("scoring=strongest", "searchNames=true", "limit=3", "accountId=a1"):
        assert part in first
    assert "since=2026-01-01" in first
    assert "alpha" not in first and "campaignId" not in first and "until" not in first
    assert "alpha=0.25" in second and "searchNames=false" in second


async def test_similar_to_idea_404_is_unavailable_or_an_unknown_idea(gub):
    """UB answers the same 404 for the flag off and for an idea that is
    missing or not visible, so the model is told to check with get_idea."""
    gub.routes[GET] = (404, {"code": "NOT_FOUND", "message": "Idea not found"})

    result = await similar_to_idea("i1")

    assert result["status"] == 404
    assert result["similarity_unavailable"] is True
    assert result["message"] == UNAVAILABLE_OR_UNKNOWN_IDEA_MESSAGE
    assert "get_idea" in result["message"] and "list_ideas" in result["message"]


async def test_similar_to_idea_503_not_configured_falls_back(gub):
    gub.routes[GET] = (503, {"code": "EMBEDDINGS_NOT_CONFIGURED"})

    result = await similar_to_idea("i1")

    assert result["similarity_unavailable"] is True
    assert result["message"] == UNAVAILABLE_MESSAGE


async def test_similar_to_idea_not_embedded_says_what_to_do(gub):
    gub.routes[GET] = (409, {"code": "IDEA_NOT_EMBEDDED"})

    result = await similar_to_idea("i1")

    assert result["error"] is True
    assert result["status"] == 409
    assert result["code"] == "IDEA_NOT_EMBEDDED"
    assert result["message"] == NOT_EMBEDDED_MESSAGE
    assert "similarity_unavailable" not in result


async def test_similar_to_idea_passes_other_errors_through(gub):
    gub.routes[GET] = (403, {})

    result = await similar_to_idea("i1")

    assert result == {
        "error": True,
        "status": 403,
        "message": _friendly_error(403, "/org/ideas/i1/similar"),
    }
