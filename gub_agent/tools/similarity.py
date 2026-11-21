"""
similarity.py — idea similarity search for the GUB agent (UB decision D40).

Tools:
  similar_ideas    — ideas like a concept, from weighted facets the model writes
  similar_to_idea  — ideas like an existing idea, from its stored embeddings

UB embeds every idea's name and every facet row separately, and scores a query
of WEIGHTED facets against them. UB never structures a question into facets:
the agent closest to the requester does, because it knows what is categorical
("an Instagram campaign") and what is the specific signal ("a huge grape rolls
down the street"). Here that agent is the executor model, so the docstrings
below are where the decomposition rules reach it.

Where the feature is off (UB's `IDEA_SIMILARITY` flag answers 404 as if the
route were not mounted) or no embedding driver is configured (503
`EMBEDDINGS_NOT_CONFIGURED`), the tool result says so plainly and sends the
model back to `list_ideas` and matching by meaning, which is how concept
questions were answered before D40. Every other error passes through as
`_client` shaped it.
"""

from __future__ import annotations

from typing import Any

from ._client import gub_get, gub_post

#: The POST route (the GET one is `/org/ideas/{id}/similar`).
SIMILAR_PATH = "/org/ideas/similar"

#: UB's code when no embedding driver is configured in this environment.
EMBEDDINGS_NOT_CONFIGURED = "EMBEDDINGS_NOT_CONFIGURED"

#: UB's code when the source idea has no current embeddings yet.
IDEA_NOT_EMBEDDED = "IDEA_NOT_EMBEDDED"

#: What the model reads when similarity search is not available here. Not a
#: claim about the ideas: the fallback still has to be run.
UNAVAILABLE_MESSAGE = (
    "Similarity search over ideas is not available here. This says nothing "
    "about which ideas exist: do not report that nothing similar was pitched. "
    "Fall back to `list_ideas` (with any account, campaign or date filters the "
    "question gives) and match the concept by meaning over the returned facets "
    "yourself."
)

#: GET /org/ideas/{id}/similar answers 404 both where the feature is off and
#: where the idea is missing, inactive or not visible, with the same code.
UNAVAILABLE_OR_UNKNOWN_IDEA_MESSAGE = (
    "Either similarity search over ideas is not available here, or this idea "
    "does not exist or is not visible to this user. Read it with `get_idea`: "
    "if it comes back, fall back to `list_ideas` and match its facets by meaning "
    "yourself; if it does not, say the idea was not found."
)

#: 409 IDEA_NOT_EMBEDDED: the idea exists but is not indexed yet.
NOT_EMBEDDED_MESSAGE = (
    "This idea is not indexed for similarity search yet (it was written in the "
    "last few minutes, or indexing has not reached it). Read it with "
    "`get_idea`, then call `similar_ideas` with its facets as the query (weight "
    "1.0 each, lower for the categorical ones); if that is not available either, "
    "fall back to `list_ideas` and match by meaning."
)


def _clean_facets(facets: Any) -> list[dict[str, Any]]:
    """The facets as UB takes them: `text` (stripped, non-empty) and `weight`
    only when the model gave one. Anything else the model put in a facet is
    dropped, because UB refuses unknown keys."""
    out: list[dict[str, Any]] = []
    for facet in facets or []:
        if isinstance(facet, str):
            facet = {"text": facet}
        if not isinstance(facet, dict):
            continue
        text = facet.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        row: dict[str, Any] = {"text": text.strip()}
        weight = facet.get("weight")
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            row["weight"] = weight
        out.append(row)
    return out


def _with_matched_on(response: Any) -> Any:
    """Each result gains `matchedOn`: the text of its first evidence entry (the
    biggest weighted contribution, as UB orders them), so the match the answer
    cites is a field of the row and not only buried in the evidence list."""
    if not isinstance(response, dict) or response.get("error"):
        return response
    for result in response.get("results") or []:
        if not isinstance(result, dict):
            continue
        evidence = result.get("evidence") or []
        top = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
        text = top.get("facet") or top.get("name")
        if isinstance(text, str) and text:
            result["matchedOn"] = text
    return response


def _is_error(response: Any, status: int) -> bool:
    return (
        isinstance(response, dict)
        and response.get("error") is True
        and response.get("status") == status
    )


def _unavailable(response: dict, message: str) -> dict:
    """The error as `_client` shaped it, its message replaced by the fallback
    instruction. Stays an error, so the evidence index never cites it."""
    return {**response, "similarity_unavailable": True, "message": message}


def _not_configured(response: Any) -> bool:
    return _is_error(response, 503) and response.get("code") == EMBEDDINGS_NOT_CONFIGURED


async def similar_ideas(
    facets: list[dict],
    scoring: str | None = None,
    alpha: float | None = None,
    search_names: bool | None = None,
    limit: int | None = None,
    account_id: str | None = None,
    campaign_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Find pitched ideas LIKE a concept, by meaning, across the whole ideas
    memory. This is the FIRST tool for a concept question: "have we ever
    pitched something like…", "have we done a giant mascot before?", "anything
    similar to this idea for Pepsi?".

    YOU break the concept into facets: short natural-language statements, one
    thought each, and give each a weight from 0 to 1 for how much a match on it
    should count:
    - CATEGORICAL facets get LOW weight (about 0.1 to 0.3): the channel, format
      or genre — "an Instagram campaign", "a social campaign", "a TV spot".
      Lots of ideas share them, so a match there is weak signal.
    - SPECIFIC facets get HIGH weight (about 0.8 to 1.0): the distinctive
      image, mechanic or twist — "a huge grape rolls down the street", "the
      truck keeps working while the town sleeps". A match there is real signal.
    Example: "have we pitched an Instagram campaign where a huge grape rolls
    down the street?" → facets=[{"text": "an Instagram campaign", "weight":
    0.2}, {"text": "a huge grape rolls down the street", "weight": 1.0}].
    Use 1 to 12 facets. Write them as statements about the idea, in the
    user's terms, not keywords. A bare one-line question is fine as one
    facet at weight 1.0.

    scoring — leave it unset (UB's default, `blend`) unless the question
    points one way:
    - "strongest" when the question is about ONE specific element ("have we
      done the rolling grape before?"): one strong facet match wins.
    - "cumulative" when it is about OVERALL likeness ("anything like this whole
      concept?"): rewards ideas that cover many of the weighted facets.
    - "blend" mixes the two; `alpha` (0 to 1, blend only) is the share of
      strongest. Never send `alpha` with strongest or cumulative.

    search_names — leave it unset (off). Set it to true ONLY when the user
    names an idea ("anything like Night Shift?"): then idea names are matched
    too. Otherwise names are noise.

    Filters (`account_id`, `campaign_id`, `since`, `until`) are optional and
    explicit: use them only when the question narrows to an account, a
    campaign or a period. Resolve ids first with `find` / `list_accounts`.

    Reading the answer:
    - `results` are best first. `score` is 0 to 1 (higher is closer); it is a
      ranking, not a verdict. Judge likeness from `evidence`, and say plainly
      when the best results are weak.
    - `evidence` lists, for each of your facets (`query` is its 0-based
      position in `facets`), the idea facet that matched it best (`facet`, or
      `name` when the match was the idea's name) and its `similarity`.
      `matchedOn` is the text of the match that counted most. CITE what
      matched: "matched on the rolling grape, not just on Instagram". A result
      that matched only your categorical facets is NOT a match for the concept;
      say so rather than presenting it as one.
    - `unembedded` counts ideas left out because they are not indexed yet.
      When it is above 0 and nothing matched well, say some ideas could not be
      searched yet rather than that the concept was never pitched.
    - `scoring` names the strategy version UB used.
    If the result says similarity search is not available, follow its
    instruction: fall back to `list_ideas` and match by meaning yourself.

    Args:
        facets: The concept as weighted facets, each {"text": str, "weight":
            float 0-1}. Low weight for the categorical, high for the specific.
        scoring: "strongest", "cumulative" or "blend" (default when unset).
        alpha: blend only — the share of strongest, 0 to 1 (UB default 0.5).
        search_names: true only when the user names an idea; default off.
        limit: Max ideas to return, 1 to 50 (UB default 10).
        account_id: UUID of an account to scope to.
        campaign_id: UUID of a campaign to scope to.
        since: Inclusive lower bound on pitchedAt (YYYY-MM-DD or ISO date-time).
        until: Inclusive upper bound on pitchedAt (YYYY-MM-DD or ISO date-time).

    Returns:
        dict { scoring, unembedded, results: [{ id, name, facets, pitchedAt,
        accountName, campaignName, awarded, score, evidence, matchedOn }] }.
        Without ideas access the list is empty (not an error).
    """
    cleaned = _clean_facets(facets)
    if not cleaned:
        return {
            "error": True,
            "status": 400,
            "message": (
                "similar_ideas needs at least one facet with text, e.g. "
                '[{"text": "a huge grape rolls down the street", "weight": 1.0}].'
            ),
        }
    body: dict[str, Any] = {"facets": cleaned}
    optional = {
        "scoring": scoring,
        "alpha": alpha,
        "searchNames": search_names,
        "limit": limit,
        "accountId": account_id,
        "campaignId": campaign_id,
        "since": since,
        "until": until,
    }
    body.update({k: v for k, v in optional.items() if v is not None})

    response = await gub_post(SIMILAR_PATH, body, tool_context)
    if _is_error(response, 404) or _not_configured(response):
        return _unavailable(response, UNAVAILABLE_MESSAGE)
    return _with_matched_on(response)


async def similar_to_idea(
    idea_id: str,
    scoring: str | None = None,
    alpha: float | None = None,
    search_names: bool | None = None,
    limit: int | None = None,
    account_id: str | None = None,
    campaign_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    tool_context: Any = None,
) -> dict:
    """
    Find pitched ideas LIKE an existing idea, using that idea's own facets as
    the query (each at weight 1). Use it when the user points at an idea you
    have already resolved to an id ("anything like the Night Shift idea?",
    "have we pitched something similar to this one elsewhere?"). Resolve the
    name with `find` first. The idea itself is never in its own results.

    When the user's interest is one PART of the idea, or the idea has
    categorical facets that would swamp the match, read it with `get_idea` and
    call `similar_ideas` instead with your own weighted facets.

    scoring: leave unset (blend) unless the question is about one specific
    element ("strongest") or overall likeness ("cumulative"). `alpha` is for
    blend only. search_names: true only when the name itself matters (then the
    idea's name is matched against other names); default off.

    Reading the answer: as `similar_ideas`. `score` ranks, `evidence` says
    which of the source idea's facets (`query` is its 0-based facet position;
    its name, with search_names, is position = its facet count) matched which
    facet of the result, `matchedOn` is the match that counted most. Cite what
    matched. `unembedded` counts ideas not indexed yet. If the result says
    similarity search is not available, or the idea is not indexed yet,
    follow its instruction.

    Args:
        idea_id: The UUID of the source idea (from find / list_ideas).
        scoring: "strongest", "cumulative" or "blend" (default when unset).
        alpha: blend only — the share of strongest, 0 to 1.
        search_names: true to also match the idea's name against other names.
        limit: Max ideas to return, 1 to 50 (UB default 10).
        account_id: UUID of an account to scope the results to.
        campaign_id: UUID of a campaign to scope the results to.
        since: Inclusive lower bound on pitchedAt (YYYY-MM-DD or ISO date-time).
        until: Inclusive upper bound on pitchedAt (YYYY-MM-DD or ISO date-time).

    Returns:
        dict { scoring, unembedded, results: [{ id, name, facets, pitchedAt,
        accountName, campaignName, awarded, score, evidence, matchedOn }] }.
    """
    response = await gub_get(
        f"/org/ideas/{idea_id}/similar",
        tool_context,
        scoring=scoring,
        alpha=None if alpha is None else str(alpha),
        searchNames=None if search_names is None else ("true" if search_names else "false"),
        limit=None if limit is None else str(limit),
        accountId=account_id,
        campaignId=campaign_id,
        since=since,
        until=until,
    )
    if _not_configured(response):
        return _unavailable(response, UNAVAILABLE_MESSAGE)
    if _is_error(response, 404):
        return _unavailable(response, UNAVAILABLE_OR_UNKNOWN_IDEA_MESSAGE)
    if _is_error(response, 409) and response.get("code") == IDEA_NOT_EMBEDDED:
        return {**response, "message": NOT_EMBEDDED_MESSAGE}
    return _with_matched_on(response)
